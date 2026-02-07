# DeePAW Documentation

This directory contains comprehensive documentation for the DeePAW project.

## 📚 Documentation Index

### Getting Started

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide for new users
  - Installation instructions
  - Basic usage examples
  - First prediction tutorial

### Tutorials

- **[external_prediction_tutorial.ipynb](external_prediction_tutorial.ipynb)** - Interactive Jupyter notebook
  - Complete prediction workflow
  - Dual model (F_nonlocal + F_local) usage
  - Single model usage
  - Comparison and analysis
  - **Recommended for external users!**

- **[NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)** - How to use the notebook
  - Launch instructions
  - Expected output
  - Customization guide
  - Troubleshooting tips

### User Guides

- **[CHGCAR_SCRIPTS_GUIDE.md](CHGCAR_SCRIPTS_GUIDE.md)** - CHGCAR file generation guide
  - Using prediction scripts
  - VASP CHGCAR format
  - Output file structure

### Advanced Features

- **[server/SERVER_GUIDE.md](server/SERVER_GUIDE.md)** - 推理服务器使用指南
  - 模型常驻 GPU，消除重复加载开销
  - Unix socket 本地调用 + HTTP API 远程调用
  - CLI 命令：`deepaw-server start/stop/status`、`deepaw-predict`
  - Python API：`DeePAWClient`
  - 后台运行、torch.compile 加速

### Technical Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project architecture
  - Directory structure
  - Module organization
  - Model architecture details
  - Data flow

### Developer Documentation

- **[archive/CLASS_RENAMING_SUMMARY.md](archive/CLASS_RENAMING_SUMMARY.md)** - Class renaming documentation
  - AtomicConfigurationModel (原子配置)
  - AtomicPotentialModel (原子势)
  - Backward compatibility
  - Migration guide

- **[archive/RENAMING_COMPLETE.md](archive/RENAMING_COMPLETE.md)** - Renaming verification
  - Test results
  - Compatibility checks
  - State dict preservation

- **[archive/TASK_COMPLETION_SUMMARY.md](archive/TASK_COMPLETION_SUMMARY.md)** - Development task history

## 🚀 Quick Navigation

### For New Users
1. Start with [QUICKSTART.md](QUICKSTART.md)
2. Try [external_prediction_tutorial.ipynb](external_prediction_tutorial.ipynb)
3. Read [CHGCAR_SCRIPTS_GUIDE.md](CHGCAR_SCRIPTS_GUIDE.md) for VASP output

### For Developers
1. Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. Check [archive/CLASS_RENAMING_SUMMARY.md](archive/CLASS_RENAMING_SUMMARY.md) for API details
3. See [archive/](archive/) for development history

### For External Integration
1. **Use [external_prediction_tutorial.ipynb](external_prediction_tutorial.ipynb)** as template
2. Adapt the prediction functions to your workflow
3. Refer to [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture details

### For High-Throughput / Production Use
1. Read [server/SERVER_GUIDE.md](server/SERVER_GUIDE.md) for persistent inference server
2. Start with `deepaw-server start`, predict with `deepaw-predict`
3. Use HTTP API for remote or cross-machine access

## 📖 Main README

For general project information, installation, and overview, see the main [README.md](../README.md) in the project root.

## 🔗 Related Resources

- **Examples**: See [examples/README.md](../examples/README.md) for detailed example tutorials
- **Scripts**: See `../deepaw/scripts/` for prediction scripts
- **Tests**: See `../tests/` for unit tests
- **Checkpoints**: See `../checkpoints/` for pretrained models

## 📝 Document Maintenance

All documentation files should be kept in this `docs/` directory to maintain a clean project structure.

### Adding New Documentation

When adding new documentation:
1. Place the file in this `docs/` directory
2. Update this README.md with a link and description
3. Use clear, descriptive filenames
4. Include appropriate section headers

### Documentation Standards

- Use Markdown format (`.md`) for text documentation
- Use Jupyter notebooks (`.ipynb`) for interactive tutorials
- Include code examples where appropriate
- Keep documentation up-to-date with code changes
- Add table of contents for long documents

## 🆘 Getting Help

If you have questions:
1. Check the relevant documentation file above
2. Look at examples in `../examples/`
3. Review the main [README.md](../README.md)
4. Open an issue on GitHub

---

**Last Updated**: 2025-02-08

