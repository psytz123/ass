import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { 
  Activity, 
  Brain, 
  Settings, 
  BarChart3, 
  MessageSquare, 
  Zap, 
  CheckCircle, 
  XCircle, 
  Clock,
  Users,
  TrendingUp,
  Send,
  Loader2
} from 'lucide-react'
import './App.css'

// API base URL - adjust for your backend
const API_BASE_URL = 'http://localhost:5000/api'

// Dashboard Component
function Dashboard() {
  const [providerStatus, setProviderStatus] = useState({})
  const [performanceMetrics, setPerformanceMetrics] = useState({})
  const [usageStats, setUsageStats] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [statusRes, metricsRes, statsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/providers/status`),
        fetch(`${API_BASE_URL}/metrics/performance`),
        fetch(`${API_BASE_URL}/usage/statistics`)
      ])

      if (statusRes.ok) setProviderStatus(await statusRes.json())
      if (metricsRes.ok) setPerformanceMetrics(await metricsRes.json())
      if (statsRes.ok) setUsageStats(await statsRes.json())
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading dashboard...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Provider Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(providerStatus).map(([provider, status]) => (
          <Card key={provider} className="relative overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium capitalize">
                  {provider.replace('_', ' ')}
                </CardTitle>
                {status.available ? (
                  <CheckCircle className="h-4 w-4 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 text-red-500" />
                )}
              </div>
            </CardHeader>
            <CardContent>
              <Badge variant={status.available ? "default" : "destructive"}>
                {status.available ? "Online" : "Offline"}
              </Badge>
              <p className="text-xs text-muted-foreground mt-2">
                {status.models?.length || 0} models available
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Performance Metrics
            </CardTitle>
            <CardDescription>
              Average response times and success rates
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(performanceMetrics).map(([provider, metrics]) => (
              <div key={provider} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium capitalize">
                    {provider.replace('_', ' ')}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {Math.round(metrics.avg_latency)}ms
                  </span>
                </div>
                <Progress value={metrics.success_rate * 100} className="h-2" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{Math.round(metrics.success_rate * 100)}% success</span>
                  <span>{metrics.total_requests} requests</span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Usage Statistics
            </CardTitle>
            <CardDescription>
              Recent activity and trends
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold">{usageStats.total_conversations || 0}</div>
                <div className="text-xs text-muted-foreground">Total Conversations</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {Math.round(usageStats.avg_processing_time_ms || 0)}ms
                </div>
                <div className="text-xs text-muted-foreground">Avg Processing Time</div>
              </div>
            </div>
            
            {usageStats.conversations_by_type && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">By Task Type</h4>
                {Object.entries(usageStats.conversations_by_type).map(([type, count]) => (
                  <div key={type} className="flex justify-between text-sm">
                    <span className="capitalize">{type.replace('_', ' ')}</span>
                    <span>{count}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// Test Interface Component
function TestInterface() {
  const [prompt, setPrompt] = useState('')
  const [taskType, setTaskType] = useState('general')
  const [requireConsensus, setRequireConsensus] = useState(false)
  const [consensusStrategy, setConsensusStrategy] = useState('similarity')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!prompt.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await fetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          task_type: taskType,
          require_consensus: requireConsensus,
          consensus_strategy: consensusStrategy,
          user_id: 'test_user'
        })
      })

      const data = await res.json()
      
      if (data.error) {
        setError(data.error)
      } else {
        setResponse(data)
      }
    } catch (err) {
      setError('Failed to connect to the backend. Make sure the Flask server is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Test AI Orchestration
          </CardTitle>
          <CardDescription>
            Test the framework with custom prompts and settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-sm font-medium">Prompt</label>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt here..."
                className="mt-1"
                rows={4}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium">Task Type</label>
                <Select value={taskType} onValueChange={setTaskType}>
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="general">General</SelectItem>
                    <SelectItem value="code_generation">Code Generation</SelectItem>
                    <SelectItem value="business_automation">Business Automation</SelectItem>
                    <SelectItem value="document_analysis">Document Analysis</SelectItem>
                    <SelectItem value="technical_analysis">Technical Analysis</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium">Consensus Strategy</label>
                <Select value={consensusStrategy} onValueChange={setConsensusStrategy}>
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="similarity">Similarity</SelectItem>
                    <SelectItem value="voting">Voting</SelectItem>
                    <SelectItem value="confidence">Confidence</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center space-x-2 mt-6">
                <input
                  type="checkbox"
                  id="consensus"
                  checked={requireConsensus}
                  onChange={(e) => setRequireConsensus(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="consensus" className="text-sm font-medium">
                  Require Consensus
                </label>
              </div>
            </div>

            <Button type="submit" disabled={loading || !prompt.trim()} className="w-full">
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="mr-2 h-4 w-4" />
                  Send Request
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {response && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Response
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-muted p-4 rounded-lg">
              <p className="whitespace-pre-wrap">{response.response}</p>
            </div>

            {response.metadata && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="font-medium">Providers Used:</span>
                  <div className="mt-1">
                    {response.metadata.providers_used?.map((provider) => (
                      <Badge key={provider} variant="outline" className="mr-1">
                        {provider}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="font-medium">Consensus Method:</span>
                  <div className="mt-1">
                    <Badge variant="secondary">
                      {response.metadata.consensus_method}
                    </Badge>
                  </div>
                </div>
                <div>
                  <span className="font-medium">Confidence:</span>
                  <div className="mt-1">
                    {Math.round(response.metadata.confidence_score * 100)}%
                  </div>
                </div>
                <div>
                  <span className="font-medium">Processing Time:</span>
                  <div className="mt-1">
                    {Math.round(response.metadata.processing_time_ms)}ms
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// Navigation Component
function Navigation() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: BarChart3 },
    { path: '/test', label: 'Test Interface', icon: MessageSquare },
  ]

  return (
    <nav className="border-b">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Brain className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold">AI Orchestration</span>
            </div>
          </div>
          
          <div className="flex space-x-4">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}

// Main App Component
function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <Navigation />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/test" element={<TestInterface />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App

