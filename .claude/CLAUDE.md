# CT Image Processing and Segmentation - Claude Code Guidelines

## Project Overview

This repository provides tools for processing, segmenting, evaluating, and classifying CT images, with a focus on atlas-based segmentation and random forest classification. The project demonstrates medical image processing workflows using Python with SimpleITK, including image registration, segmentation evaluation, and machine learning classification for anatomical structure identification.

## Development Environment

**Operating System**: Windows 11
**Shell**: Git Bash / PowerShell / Command Prompt
**Important**: Always use Windows-compatible commands:
- Use `dir` instead of `ls` for Command Prompt
- Use PowerShell commands when appropriate
- File paths use backslashes (`\`) in Windows
- Use `python -m http.server` for local development server
- Git Bash provides Unix-like commands but context should be Windows-aware

## Development Guidelines

### Code Quality
- Follow Python PEP 8 style guidelines
- Use meaningful variable and function names
- Implement proper error handling and logging
- Add comprehensive docstrings for functions and classes
- Use type hints where appropriate
- Maintain clean, readable code
- Follow language-specific best practices

### Security
- No sensitive information in the codebase
- Use HTTPS for all external resources
- Regular dependency updates
- Follow security best practices for the specific technology stack

### Medical Image Processing Specific Guidelines
- Document image processing pipelines and parameters clearly
- Implement proper medical image format handling (NIfTI, DICOM, etc.)
- Use appropriate evaluation metrics (Dice, Jaccard, Hausdorff distance)
- Ensure reproducibility with proper random seeds
- Implement proper image registration and segmentation workflows
- Use validated medical image processing libraries (SimpleITK, ITK, etc.)
- Document coordinate systems and image orientations
- Handle edge cases in medical image data appropriately

## Learning and Communication
- Always explain coding actions and decisions to help the user learn
- Describe why specific approaches or technologies are chosen
- Explain the purpose and functionality of code changes
- Provide context about best practices and coding patterns used
- Provide detailed explanations in the console when performing tasks, as many concepts may be new to the user