# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Iris Memory Plugin is a three-layer memory system plugin for AstrBot that provides intelligent memory management. It implements a companion-memory framework with Working, Episodic, and Semantic memory layers.

## Development Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_iris_memory.py -v

# Run single test
python -m pytest tests/test_iris_memory.py::test_memory_creation -v
```

## Architecture

```
iris_memory/
├── models/           # Data models (Memory, UserPersona, EmotionState)
├── storage/          # Persistence layer (Chroma DB, cache, sessions)
├── capture/          # Memory capture pipeline (triggers, sensitivity)
├── retrieval/        # Memory retrieval (hybrid search, reranking)
├── analysis/         # Analysis tools (emotion, RIF scoring, entities)
├── utils/            # Utilities (logger, hooks, token management)
└── core/             # Type definitions and configuration
```

### Core Data Flow

**Capture Pipeline:** User Message → Trigger Detection → Emotion Analysis → Sensitivity Detection → Quality Assessment → RIF Scoring → Chroma Storage

**Retrieval Pipeline:** User Query → Retrieval Router → Hybrid Search (vector + RIF + time + emotion) →情感过滤 →重排序 →注入LLM上下文

### Key Classes

- `Star` (main.py): Main plugin entry point extending AstrBot's Star class
- `Memory`: Core data model with to_dict/from_dict serialization
- `SessionManager`: User/group session isolation using `user_id:group_id` keys
- `WorkingMemoryCache`: LRU cache for session-scoped temporary memory
- `EmotionAnalyzer`: Hybrid emotion detection (lexicon + rules)

### Session Key Format

Sessions use format `user_id:group_id` or `user_id:private` for private chats. This enables complete isolation between private and group conversations.
