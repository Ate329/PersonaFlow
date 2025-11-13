# Changelog

All notable changes to PersonaFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-11-13

### Fixed
- **Critical:** Fixed IndexError in memory summarization when `memories_to_summarize` list is empty
- **Critical:** Fixed incorrect exception type in JSON serializer (`TypeError` instead of `JSONDecodeError` for `json.dump()`)
- Fixed overly restrictive memory content validation - now accepts any non-empty dictionary structure
- Fixed memory config validation to properly accept partial configurations with defaults
- Improved error handling in `add_interaction()` with explicit KeyError for missing characters
- Fixed character deserialization with safe defaults for optional fields

### Added
- Input validation for empty character names and prompts
- Input validation for negative memory limits
- Input validation for invalid memory config values (≤ 0)
- Input validation for empty template names and content
- Validation for memory manager character names
- Comprehensive `.gitignore` covering build artifacts, IDEs, OS files, and more
- `pytest.ini` with test configuration and coverage settings
- `MANIFEST.in` for proper package distribution
- CI/CD workflow (`.github/workflows/ci.yml`) for automated testing on multiple platforms
- Automated linting and type checking in CI pipeline

### Changed
- Improved character deserialization to handle missing optional fields gracefully
- Better error messages throughout the codebase for easier debugging
- Enhanced memory config validation to check field types without requiring all fields

### Development
- Fixed file handle leak in `setup.py` by using `pathlib` for reading README
- Removed `dist/` folder from git tracking (now in `.gitignore`)
- Created separate `nix-devenv` branch for Nix development environment files
- Added proper package metadata in `setup.py`

## [0.1.2] - 2024-10-27

### Enhancements

**Selective Broadcasting**
- Enhanced `PersonaSystem.broadcast_interaction()` with selective broadcasting capabilities
- Added ability to target specific characters with `broadcast_to` parameter
- Added ability to exclude specific characters with `exclude_characters` parameter

### Example Usage
```python
# Broadcast to specific characters
system.broadcast_interaction(
    content={"message": "Hello targeted characters!"},
    broadcast_to=["character1", "character2"]
)

# Broadcast to all except excluded characters
system.broadcast_interaction(
    content={"message": "Hello everyone except you!"},
    exclude_characters=["character3"]
)
```

## [0.1.1] - 2024-10-24

### Added
- Merge branch 'main' of https://github.com/Ate329/PersonaFlow

## [0.1.0] - 2024-10-24

### Added
- Initial release of PersonaFlow
- Core character management system
- Memory management with automatic summarization
- Multi-character interaction system
- Prompt template management
- Serialization utilities
- Logger utilities
- Basic test suite

