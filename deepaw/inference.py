"""
DeePAW InferenceEngine: 模型常驻显存的推理引擎

用法:
    from deepaw.inference import InferenceEngine

    # 初始化一次（加载模型到 GPU，常驻显存）
    engine = InferenceEngine()

    # 之后反复调用 predict，无需重新加载模型
    result = engine.predict(db_path="examples/hfo2_chgd.db", db_id=1)
    result = engine.predict(atoms=my_atoms, grid_shape=(80, 80, 80))

    # 也可以直接输出 CHGCAR 文件
    engine.predict_and_write_chgcar(db_path="examples/hfo2_chgd.db", db_id=1, output_path="CHGCAR")
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from ase.db import connect

from deepaw.data.chgcar_writer import GraphConstructor, get_grid_centers
from deepaw.config import get_model_config, get_checkpoint_path


class InferenceEngine:
    """模型常驻显存的推理引擎。

    初始化时加载模型到 GPU 并保持常驻，后续每次 predict() 直接推理，
    省去重复的模型初始化和权重加载开销。

    Args:
        checkpoint_dir: 权重文件目录，默认为项目根目录下的 checkpoints/
        device: 推理设备，默认自动选择 cuda/cpu
        cutoff: 邻居搜索截断半径 (Angstrom)
        data_batch_size: 每批 probe 点数量，受 GPU 显存限制 (24GB GPU 建议 3000)
        use_dual_model: 是否使用 F_local 修正模型 (更高精度)
        use_compile: 是否使用 torch.compile 加速 probe_model (约 1.14x 加速)
        use_gpu_graph: 是否使用 GPU 加速图构建 (None=自动, True=强制, False=禁用)
    """

    def __init__(
        self,
        checkpoint_dir=None,
        device=None,
        cutoff=4.0,
        data_batch_size=3000,
        use_dual_model=True,
        use_compile=True,
        use_gpu_graph=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.cutoff = cutoff
        self.data_batch_size = data_batch_size
        self.use_dual_model = use_dual_model
        self.use_compile = use_compile and (device != "cpu")
        self.use_gpu_graph = use_gpu_graph

        # 确定 checkpoint 目录
        if checkpoint_dir is None:
            deepaw_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            checkpoint_dir = os.path.join(deepaw_root, "checkpoints")
        self.checkpoint_dir = checkpoint_dir

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载模型权重到目标设备并保持常驻。"""
        import e3nn

        # 使用 eager 模式替代 TorchScript，允许 torch.compile 优化
        if self.use_compile:
            e3nn.set_optimization_defaults(jit_mode='eager')

        # 延迟导入，确保 e3nn 优化设置已生效
        from deepaw.models.f_nonlocal import F_nonlocal
        from deepaw.models.f_local import F_local

        f_nonlocal_config = get_model_config("f_nonlocal")
        self.f_nonlocal = F_nonlocal(**f_nonlocal_config).to(self.device)

        path_nonlocal = os.path.join(
            self.checkpoint_dir, os.path.basename(get_checkpoint_path("f_nonlocal"))
        )
        if not os.path.exists(path_nonlocal):
            raise FileNotFoundError(f"F_nonlocal 权重不存在: {path_nonlocal}")

        # eager 模式下 state_dict 不含 _w3j Clebsch-Gordan 系数，需 strict=False
        strict = not self.use_compile
        self.f_nonlocal.load_state_dict(
            torch.load(path_nonlocal, map_location=self.device), strict=strict
        )
        self.f_nonlocal.eval()

        # torch.compile 加速 probe_model（占 GPU 推理时间 ~89%）
        if self.use_compile:
            self.f_nonlocal.probe_model = torch.compile(
                self.f_nonlocal.probe_model, mode='default', fullgraph=False
            )

        if self.use_dual_model:
            f_local_config = get_model_config("f_local")
            self.f_local = F_local(**f_local_config).to(self.device)

            path_local = os.path.join(
                self.checkpoint_dir,
                os.path.basename(get_checkpoint_path("f_local")),
            )
            if not os.path.exists(path_local):
                raise FileNotFoundError(f"F_local 权重不存在: {path_local}")
            self.f_local.load_state_dict(
                torch.load(path_local, map_location=self.device)
            )
            # F_local 不能使用 eval()（KAN 网络 bug）
        else:
            self.f_local = None

        self._graph_constructor = GraphConstructor(
            cutoff=self.cutoff, num_probes=None, disable_pbc=False,
            use_gpu_graph=self.use_gpu_graph, device=self.device,
        )

    def predict(self, db_path=None, db_id=None, atoms=None, grid_shape=None):
        """预测电荷密度。

        两种输入方式（二选一）：
          1. db_path + db_id: 从 ASE 数据库读取结构
          2. atoms + grid_shape: 直接传入 ASE Atoms 对象和网格尺寸

        Args:
            db_path: ASE 数据库路径
            db_id: 数据库中的结构 ID（从 1 开始）
            atoms: ASE Atoms 对象
            grid_shape: 网格尺寸 (nx, ny, nz) 的元组

        Returns:
            dict: {
                "density_3d": np.ndarray (nx, ny, nz) 三维电荷密度,
                "density_flat": np.ndarray (nx*ny*nz,) 展平电荷密度,
                "atoms": ASE Atoms 对象,
                "grid_shape": (nx, ny, nz),
            }
        """
        atoms, nx, ny, nz = self._resolve_input(db_path, db_id, atoms, grid_shape)

        atoms = atoms.copy()
        atoms.set_pbc(True)

        # 生成网格探针点
        probe_pos = get_grid_centers(atoms, nx, ny, nz)
        num_positions = nx * ny * nz

        # 预计算原子不变数据（单结构只算一次）
        atom_cache = self._graph_constructor.precompute_atom_data(atoms)

        all_predictions = []
        batch_size = self.data_batch_size

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, num_positions, batch_size),
                desc="推理中",
                total=(num_positions + batch_size - 1) // batch_size,
                leave=False,
            ):
                end_idx = min(start_idx + batch_size, num_positions)

                unravel_indices = np.unravel_index(
                    np.arange(start_idx, end_idx), (nx, ny, nz)
                )
                probe_pos_batch = probe_pos[unravel_indices]
                # inference=True 时不创建 probe_target
                dummy_density = np.zeros(end_idx - start_idx)

                graph_dict = self._graph_constructor.build_graph_with_cache(
                    density=dummy_density,
                    grid_pos=probe_pos_batch,
                    cache=atom_cache,
                    inference=True,
                )

                # 添加 batch 维度并传输到 GPU
                batch = {
                    k: v.unsqueeze(0).to(self.device, non_blocking=True)
                    for k, v in graph_dict.items()
                }

                # F_nonlocal 前向传播
                output_nonlocal, node_rep = self.f_nonlocal(batch)
                output_nonlocal = output_nonlocal.view(-1)

                # F_local 修正
                if self.f_local is not None:
                    correction, _ = self.f_local(None, node_rep)
                    correction = correction.view(-1)
                    output_final = output_nonlocal + correction
                else:
                    output_final = output_nonlocal

                all_predictions.append(output_final.detach().cpu())

        density_flat = torch.cat(all_predictions, dim=0).numpy()
        density_3d = density_flat.reshape(nx, ny, nz)

        return {
            "density_3d": density_3d,
            "density_flat": density_flat,
            "atoms": atoms,
            "grid_shape": (nx, ny, nz),
        }

    def predict_and_write_chgcar(
        self, output_path, db_path=None, db_id=None, atoms=None, grid_shape=None
    ):
        """预测电荷密度并写入 CHGCAR 文件。

        Args:
            output_path: 输出文件路径（不含扩展名，会自动加 .vasp）
            db_path, db_id, atoms, grid_shape: 同 predict()

        Returns:
            dict: 同 predict() 的返回值
        """
        from ase.calculators.vasp import VaspChargeDensity

        result = self.predict(
            db_path=db_path, db_id=db_id, atoms=atoms, grid_shape=grid_shape
        )

        vcd = VaspChargeDensity(filename=None)
        vcd.atoms.append(result["atoms"])
        vcd.chg.append(result["density_3d"])

        if not output_path.endswith(".vasp"):
            output_path = f"{output_path}.vasp"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        vcd.write(output_path, format="chgcar")

        return result

    def _resolve_input(self, db_path, db_id, atoms, grid_shape):
        """解析输入参数，返回 (atoms, nx, ny, nz)。"""
        if db_path is not None and db_id is not None:
            db = connect(db_path)
            row = db.get(id=db_id)
            atoms = row.toatoms()
            nx, ny, nz = row.data["nx"], row.data["ny"], row.data["nz"]
            return atoms, nx, ny, nz

        if atoms is not None and grid_shape is not None:
            nx, ny, nz = grid_shape
            return atoms, nx, ny, nz

        raise ValueError(
            "必须提供 (db_path + db_id) 或 (atoms + grid_shape) 之一"
        )
