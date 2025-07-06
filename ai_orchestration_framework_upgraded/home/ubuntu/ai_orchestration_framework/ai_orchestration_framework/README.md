# AI Orchestration Framework

A powerful, extensible Python framework for building AI-powered applications with multi-model integration, dynamic routing, consensus mechanisms, and self-improvement capabilities.

## 🚀 Overview

The AI Orchestration Framework enables developers to build sophisticated AI applications that can:
- **Integrate multiple AI models** (OpenAI, Anthropic, Google, AbacusAI) seamlessly
- **Dynamically route tasks** to the most suitable model based on complexity and performance
- **Find consensus** across multiple model responses using pluggable algorithms
- **Track performance** through comprehensive analytics and monitoring
- **Scale efficiently** with async processing and intelligent caching

Perfect for building code generation tools, business automation agents, and complex AI workflows.

## ✨ Key Features

### 🔗 Multi-Model Integration
- **Universal API**: Single interface for OpenAI, Anthropic, Google, and AbacusAI models
- **Async Processing**: Parallel model calls for improved performance
- **Auto-Failover**: Automatic fallback to alternative models
- **Performance Tracking**: Built-in metrics collection and analysis

### 🧠 Intelligent Task Routing
- **Complexity Analysis**: Automatic task complexity evaluation
- **Performance-Based Routing**: Routes tasks based on historical model performance
- **Pluggable Evaluators**: Custom complexity evaluation algorithms
- **Dynamic Optimization**: Continuously improves routing decisions

### 🤝 Consensus Mechanisms
- **Multiple Strategies**: Similarity-based, voting, confidence-weighted consensus
- **Pluggable Architecture**: Easy to add custom consensus algorithms
- **Quality Metrics**: Confidence scoring and consensus quality assessment
- **Configurable Thresholds**: Customizable consensus requirements

### 📊 Memory & Analytics
- **Conversation History**: Persistent storage of all interactions
- **Performance Tracking**: Detailed metrics on model performance
- **Embedding Cache**: Efficient similarity search and caching
- **Data Export**: Easy access to historical data for analysis

### 🎨 Modern Web Interface
- **React Dashboard**: Real-time performance monitoring
- **Interactive Testing**: Test framework with custom prompts and settings
- **Dark Theme**: Professional, modern UI with Tailwind CSS
- **Responsive Design**: Works on desktop and mobile devices

## 📦 Installation

### Prerequisites
- Python 3.11+
- Node.js 20+
- npm or pnpm for package management

### Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai_orchestration_framework

# Set up Python virtual environment
cd backend
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend/ai-orchestration-dashboard

# Install dependencies
npm install

# Start development server
npm run dev -- --host
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# AI Model API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ABACUS_AI_API_KEY=your_abacus_ai_api_key_here
ABACUS_AI_DEPLOYMENT_ID=your_abacus_ai_deployment_id_here

# Database Configuration
DATABASE_PATH=ai_framework.db

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your_secret_key_here
```

## 🚀 Quick Start

### 1. Start the Backend

```bash
cd backend
source venv/bin/activate
python app.py
```

The Flask API will be available at `http://localhost:5000`

### 2. Start the Frontend

```bash
cd frontend/ai-orchestration-dashboard
npm run dev -- --host
```

The React dashboard will be available at `http://localhost:5173`

### 3. Test the Framework

1. Open the dashboard in your browser
2. Navigate to the "Test Interface" tab
3. Enter a test prompt
4. Configure task type and consensus settings
5. Click "Send Request" to test the framework

## 📖 API Reference

### Core Endpoints

#### Process Request
```http
POST /api/process
Content-Type: application/json

{
  "prompt": "Your prompt here",
  "task_type": "general",
  "require_consensus": false,
  "consensus_strategy": "similarity",
  "user_id": "user123"
}
```

#### Get Provider Status
```http
GET /api/providers/status
```

#### Get Performance Metrics
```http
GET /api/metrics/performance?time_window_hours=24
```

#### Get Usage Statistics
```http
GET /api/usage/statistics?time_window_hours=24
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                        │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │    Dashboard    │    │    Test Interface               │ │
│  │   Monitoring    │    │  (Interactive Testing)         │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │ HTTP/REST API
┌─────────────────────────────────────────────────────────────┐
│                 Flask Backend (Python)                     │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Task Router    │  │ Consensus Engine│  │   Memory    │ │
│  │ (Complexity     │  │ (Pluggable      │  │  Manager    │ │
│  │  Evaluation)    │  │  Algorithms)    │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                             │                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Model Registry & Connectors                 │ │
│  │  [OpenAI] [Anthropic] [Google] [AbacusAI]             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│              External AI Model APIs                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   OpenAI    │  │   Google    │  │     Anthropic       │ │
│  │     API     │  │     AI      │  │       API           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Testing

### Running Backend Tests

```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

### Test Coverage

The framework includes comprehensive tests for:
- Model connectors and registry
- Task routing and complexity evaluation
- Consensus mechanisms
- Memory management
- API endpoints

## 🔧 Development

### Project Structure

```
ai_orchestration_framework/
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_connectors.py    # AI model integrations
│   │   ├── task_router.py         # Intelligent task routing
│   │   ├── consensus.py           # Consensus algorithms
│   │   └── memory.py              # Data persistence
│   ├── tests/
│   │   └── test_basic.py          # Test suite
│   ├── app.py                     # Flask application
│   ├── requirements.txt           # Python dependencies
│   └── .env.example               # Environment variables template
├── frontend/
│   └── ai-orchestration-dashboard/
│       ├── src/
│       │   ├── App.jsx            # Main React application
│       │   └── App.css            # Tailwind CSS styles
│       ├── index.html             # HTML entry point
│       └── package.json           # Node.js dependencies
└── README.md                      # This file
```

### Adding New Model Providers

1. Create a new connector class inheriting from `BaseModelConnector`
2. Implement required methods: `generate()`, `validate_connection()`, `get_supported_models()`
3. Register the connector in the `AIOrchestrationFramework` initialization
4. Add API key configuration to environment variables

### Adding New Consensus Strategies

1. Create a new strategy class inheriting from `ConsensusStrategy`
2. Implement `find_consensus()` and `calculate_confidence()` methods
3. Register the strategy in the `ConsensusManager`

## 📊 Performance Monitoring

The framework automatically tracks:
- **Response Latency**: Time to generate responses
- **Token Usage**: Token consumption per provider
- **Success Rates**: Percentage of successful requests
- **Consensus Quality**: Agreement levels between models
- **Provider Performance**: Comparative analysis across providers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI, Anthropic, Google, and AbacusAI for their excellent AI APIs
- The React and Flask communities for their robust frameworks
- Contributors and beta testers who helped shape this framework

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation for detailed guides
- Review the test suite for usage examples

---

**Built with ❤️ for the AI development community**

