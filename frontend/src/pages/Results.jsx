import { useState, useEffect, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowLeft,
  Download,
  RefreshCw,
  Info,
  ChevronDown,
  ChevronUp,
  Eye,
  Code,
  BarChart3,
  Lock,
  Bug,
  Zap,
  Bot,
  Sparkles,
  Send,
  Copy,
  Check,
  Cpu
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
  Tooltip
} from 'recharts'
import { Button, Card, CardContent, Badge, Progress } from '../components/ui'
import { getAIHelp } from '../services/api'

// Risk level colors and icons
const riskConfig = {
  Safe: { color: 'risk-safe', bgColor: 'bg-risk-safe', icon: CheckCircle, gradient: 'from-green-500 to-emerald-500' },
  Low: { color: 'risk-low', bgColor: 'bg-risk-low', icon: Info, gradient: 'from-lime-500 to-green-500' },
  Medium: { color: 'risk-medium', bgColor: 'bg-risk-medium', icon: AlertTriangle, gradient: 'from-yellow-500 to-orange-500' },
  High: { color: 'risk-high', bgColor: 'bg-risk-high', icon: AlertTriangle, gradient: 'from-orange-500 to-red-500' },
  Critical: { color: 'risk-critical', bgColor: 'bg-risk-critical', icon: XCircle, gradient: 'from-red-500 to-rose-600' },
}

// Vulnerability type icons
const vulnIcons = {
  'Arithmetic': BarChart3,
  'Access Control': Lock,
  'Unchecked Calls': AlertTriangle,
  'Reentrancy': Bug,
}

// Vuln type → API issue key
const vulnToIssueKey = {
  'Arithmetic': 'arithmetic',
  'Access Control': 'access_control',
  'Unchecked Calls': 'unchecked_calls',
  'Reentrancy': 'reentrancy',
}

// Simple markdown renderer for AI responses
function MarkdownBlock({ content }) {
  const lines = content.split('\n')
  const elements = []
  let inCodeBlock = false
  let codeLines = []
  let codeLang = ''

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <div key={i} className="relative group my-3">
            <div className="flex items-center justify-between bg-dark-900 px-3 py-1.5 rounded-t-lg border border-dark-700">
              <span className="text-xs text-dark-400 font-mono">{codeLang || 'solidity'}</span>
            </div>
            <pre className="bg-dark-950 border border-t-0 border-dark-700 rounded-b-lg p-4 overflow-x-auto">
              <code className="text-sm text-green-300 font-mono">{codeLines.join('\n')}</code>
            </pre>
          </div>
        )
        codeLines = []
        codeLang = ''
        inCodeBlock = false
      } else {
        inCodeBlock = true
        codeLang = line.slice(3).trim()
      }
    } else if (inCodeBlock) {
      codeLines.push(line)
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={i} className="text-lg font-bold text-primary-300 mt-4 mb-2">{line.slice(3)}</h2>)
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={i} className="text-base font-semibold text-dark-200 mt-3 mb-1">{line.slice(4)}</h3>)
    } else if (line.startsWith('**') && line.endsWith('**')) {
      elements.push(<p key={i} className="font-semibold text-dark-100 my-1">{line.slice(2, -2)}</p>)
    } else if (line.startsWith('- ')) {
      elements.push(
        <div key={i} className="flex items-start gap-2 my-1">
          <span className="text-primary-400 mt-1 flex-shrink-0">•</span>
          <span className="text-dark-300 text-sm">{line.slice(2)}</span>
        </div>
      )
    } else if (line.trim() === '') {
      elements.push(<div key={i} className="my-1" />)
    } else {
      elements.push(<p key={i} className="text-dark-300 text-sm my-1">{line}</p>)
    }
  }

  return <div className="space-y-0.5">{elements}</div>
}

// Animated risk gauge
function RiskGauge({ score, level }) {
  const config = riskConfig[level] || riskConfig.Medium
  const Icon = config.icon
  const rotation = score * 180

  return (
    <div className="relative w-64 h-32 mx-auto">
      <svg className="w-full h-full" viewBox="0 0 200 100">
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset="25%" stopColor="#84cc16" />
            <stop offset="50%" stopColor="#eab308" />
            <stop offset="75%" stopColor="#f97316" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        <path d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="12" strokeLinecap="round" />
        <motion.path
          d="M 20 90 A 80 80 0 0 1 180 90"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="12"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: score }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
        />
        <motion.g
          initial={{ rotate: 0 }}
          animate={{ rotate: rotation }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
          style={{ transformOrigin: '100px 90px' }}
        >
          <line x1="100" y1="90" x2="100" y2="30" stroke="white" strokeWidth="3" strokeLinecap="round" />
          <circle cx="100" cy="90" r="8" fill="white" />
        </motion.g>
      </svg>
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, type: 'spring' }}
          className="flex items-center justify-center gap-2"
        >
          <Icon className={`w-6 h-6 text-${config.color}`} />
          <span className={`text-3xl font-bold text-${config.color}`}>
            {(score * 100).toFixed(0)}%
          </span>
        </motion.div>
        <Badge variant={level.toLowerCase()} className="mt-2">
          {level} Risk
        </Badge>
      </div>
    </div>
  )
}

// Vulnerability card component
function VulnerabilityCard({ vuln, index, code, onAIHelp }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const Icon = vulnIcons[vuln.type] || AlertTriangle

  const getRiskVariant = (prob) => {
    if (prob >= 0.8) return 'critical'
    if (prob >= 0.6) return 'high'
    if (prob >= 0.4) return 'medium'
    if (prob >= 0.2) return 'low'
    return 'safe'
  }

  const variant = getRiskVariant(vuln.probability)
  const borderColor = {
    safe: 'border-l-green-500',
    low: 'border-l-lime-500',
    medium: 'border-l-yellow-500',
    high: 'border-l-orange-500',
    critical: 'border-l-red-500',
  }[variant]

  const iconBg = {
    safe: 'bg-green-500/20',
    low: 'bg-lime-500/20',
    medium: 'bg-yellow-500/20',
    high: 'bg-orange-500/20',
    critical: 'bg-red-500/20',
  }[variant]

  const iconColor = {
    safe: 'text-green-400',
    low: 'text-lime-400',
    medium: 'text-yellow-400',
    high: 'text-orange-400',
    critical: 'text-red-400',
  }[variant]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
    >
      <Card hover={false} className={`border-l-4 ${borderColor}`}>
        <CardContent>
          <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${iconBg}`}>
                <Icon className={`w-5 h-5 ${iconColor}`} />
              </div>
              <div>
                <h4 className="font-semibold text-dark-100">{vuln.type}</h4>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant={variant} size="sm">{vuln.confidence} Confidence</Badge>
                  {vuln.affected_lines.length > 0 && (
                    <span className="text-xs text-dark-400">Lines: {vuln.affected_lines.join(', ')}</span>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-2xl font-bold text-dark-100">{(vuln.probability * 100).toFixed(0)}%</div>
                <div className="text-xs text-dark-400">Probability</div>
              </div>
              {isExpanded ? <ChevronUp className="w-5 h-5 text-dark-400" /> : <ChevronDown className="w-5 h-5 text-dark-400" />}
            </div>
          </div>

          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="mt-4 pt-4 border-t border-dark-700/50">
                  <p className="text-dark-400 text-sm mb-4">{vuln.description}</p>
                  <Progress value={vuln.probability * 100} variant={variant} showLabel label="Detection Confidence" />

                  {/* AI Help button - only for detected vulns */}
                  {vuln.probability > 0.4 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="mt-4"
                    >
                      <button
                        onClick={(e) => { e.stopPropagation(); onAIHelp(vuln.type, code) }}
                        className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white rounded-lg text-sm font-medium transition-all duration-200 group"
                      >
                        <Bot className="w-4 h-4 group-hover:animate-pulse" />
                        <Sparkles className="w-3 h-3" />
                        Get AI Fix
                      </button>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  )
}

// Code with attention heatmap
function CodeHeatmap({ code, lineRisks }) {
  const lines = code.split('\n')

  const getLineRiskClass = (risk) => {
    if (risk >= 0.8) return 'line-risk-critical'
    if (risk >= 0.6) return 'line-risk-high'
    if (risk >= 0.4) return 'line-risk-medium'
    if (risk >= 0.2) return 'line-risk-low'
    return ''
  }

  return (
    <div className="code-editor overflow-auto max-h-[600px]">
      {lines.map((line, index) => {
        const lineRisk = lineRisks[index] || { risk_score: 0, is_vulnerable: false }
        const riskClass = getLineRiskClass(lineRisk.risk_score)

        return (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.01 }}
            className={`code-line ${riskClass}`}
          >
            <span className="code-line-number">{index + 1}</span>
            <code className="code-line-content">{line || ' '}</code>
            {lineRisk.is_vulnerable && (
              <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="flex-shrink-0">
                <AlertTriangle className="w-4 h-4 text-red-400" />
              </motion.div>
            )}
          </motion.div>
        )
      })}
    </div>
  )
}

// AI Assistant Panel
function AIAssistantPanel({ code, vulnerabilities }) {
  const [selectedVuln, setSelectedVuln] = useState(null)
  const [aiResponse, setAiResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  const detectedVulns = vulnerabilities.filter(v => v.probability > 0.4)

  const handleGetHelp = useCallback(async (vulnType) => {
    const issueKey = vulnToIssueKey[vulnType] || vulnType.toLowerCase().replace(' ', '_')
    setSelectedVuln(vulnType)
    setLoading(true)
    setError(null)
    setAiResponse(null)

    try {
      const result = await getAIHelp(code, issueKey)
      setAiResponse(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [code])

  const handleCopy = () => {
    if (aiResponse?.response) {
      navigator.clipboard.writeText(aiResponse.response)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card hover={false}>
        <CardContent>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-dark-100 flex items-center gap-2">
                AI Security Assistant
                <Sparkles className="w-4 h-4 text-violet-400" />
              </h3>
              <p className="text-xs text-dark-400">Powered by Google Gemini 1.5 Flash</p>
            </div>
          </div>

          {detectedVulns.length === 0 ? (
            <div className="text-center py-8 text-dark-400">
              <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-400 opacity-50" />
              <p>No significant vulnerabilities detected.</p>
              <p className="text-xs mt-1">Great job writing secure code!</p>
            </div>
          ) : (
            <div>
              <p className="text-dark-400 text-sm mb-3">
                Select a vulnerability to get AI-powered explanation and fix:
              </p>
              <div className="space-y-2">
                {detectedVulns.map((vuln) => {
                  const Icon = vulnIcons[vuln.type] || AlertTriangle
                  const isSelected = selectedVuln === vuln.type
                  return (
                    <motion.button
                      key={vuln.type}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => handleGetHelp(vuln.type)}
                      disabled={loading}
                      className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all duration-200 ${
                        isSelected
                          ? 'border-violet-500/50 bg-violet-500/10'
                          : 'border-dark-700 bg-dark-800/50 hover:border-dark-600'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Icon className={`w-4 h-4 ${isSelected ? 'text-violet-400' : 'text-dark-400'}`} />
                        <span className={`text-sm font-medium ${isSelected ? 'text-violet-300' : 'text-dark-200'}`}>
                          {vuln.type}
                        </span>
                        <Badge variant={vuln.probability >= 0.6 ? 'critical' : 'medium'} size="sm">
                          {(vuln.probability * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        {loading && isSelected && (
                          <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                        )}
                        <Send className="w-3 h-3 text-dark-400" />
                      </div>
                    </motion.button>
                  )
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Loading state */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <Card hover={false}>
              <CardContent>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-violet-400" />
                  </div>
                  <span className="text-dark-300 text-sm">Analyzing {selectedVuln}...</span>
                </div>
                <div className="space-y-2">
                  {[100, 80, 60, 90].map((w, i) => (
                    <motion.div
                      key={i}
                      className="h-3 bg-dark-700 rounded"
                      style={{ width: `${w}%` }}
                      animate={{ opacity: [0.4, 1, 0.4] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Error state */}
        {error && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Card hover={false} className="border-red-500/30">
              <CardContent>
                <div className="flex items-center gap-2 text-red-400 text-sm">
                  <XCircle className="w-4 h-4" />
                  <span>{error}</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* AI Response */}
        {aiResponse && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <Card hover={false} className="border-violet-500/20">
              <CardContent>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                      <Cpu className="w-3.5 h-3.5 text-white" />
                    </div>
                    <div>
                      <span className="text-sm font-medium text-dark-200">{selectedVuln} Analysis</span>
                      <div className="flex items-center gap-1 mt-0.5">
                        <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                        <span className="text-xs text-dark-400">{aiResponse.model}</span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-dark-700 hover:bg-dark-600 text-dark-300 text-xs transition-colors"
                  >
                    {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>

                <div className="bg-dark-900/50 rounded-xl p-4 border border-dark-700/50 max-h-[500px] overflow-y-auto">
                  <MarkdownBlock content={aiResponse.response} />
                </div>

                {aiResponse.status === 'fallback' && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-dark-400">
                    <Info className="w-3 h-3" />
                    <span>Using static suggestions. Set GEMINI_API_KEY for AI-powered analysis.</span>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Results() {
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [code, setCode] = useState('')
  const [activeTab, setActiveTab] = useState('overview')
  const [aiVuln, setAiVuln] = useState(null)

  useEffect(() => {
    const storedResult = sessionStorage.getItem('analysisResult')
    const storedCode = sessionStorage.getItem('analyzedCode')

    if (storedResult && storedCode) {
      setResult(JSON.parse(storedResult))
      setCode(storedCode)
    } else {
      navigate('/analyze')
    }
  }, [navigate])


  if (!result) {
    return (
      <div className="pt-24 pb-12 min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500 mx-auto mb-4" />
          <p className="text-dark-400">Loading results...</p>
        </div>
      </div>
    )
  }

  const chartData = result.vulnerabilities.map((v) => ({
    name: v.type.split(' ')[0],
    probability: v.probability * 100,
    fill: v.probability >= 0.6 ? '#ef4444' : v.probability >= 0.4 ? '#f97316' : v.probability >= 0.2 ? '#eab308' : '#22c55e'
  }))

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'code', label: 'Code Analysis', icon: Code },
    { id: 'ai', label: 'AI Assistant', icon: Bot },
  ]

  return (
    <div className="pt-24 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <div className="flex items-center gap-4">
            <Link to="/analyze">
              <Button variant="ghost" icon={ArrowLeft}>Back</Button>
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-dark-100">Analysis Results</h1>
              <p className="text-dark-400 mt-1">Hybrid GNN + GraphCodeBERT detection complete</p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" icon={RefreshCw} onClick={() => navigate('/analyze')}>
              New Analysis
            </Button>
            <Button
              variant="secondary"
              icon={Download}
              onClick={() => {
                const data = JSON.stringify({ result, code }, null, 2)
                const blob = new Blob([data], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'vulnerability-report.json'
                a.click()
              }}
            >
              Export
            </Button>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1 space-y-6"
          >
            {/* Risk Score Card */}
            <Card hover={false}>
              <CardContent className="text-center py-8">
                <h3 className="text-lg font-semibold text-dark-200 mb-6">Overall Risk Score</h3>
                <RiskGauge score={result.overall_risk_score} level={result.risk_level} />
              </CardContent>
            </Card>

            {/* Probability Chart */}
            <Card hover={false}>
              <CardContent>
                <h3 className="text-lg font-semibold text-dark-200 mb-4">Vulnerability Probabilities</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical">
                      <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 12 }} />
                      <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={80} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                        labelStyle={{ color: '#f1f5f9' }}
                      />
                      <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={index} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Summary */}
            <Card hover={false}>
              <CardContent>
                <h3 className="text-lg font-semibold text-dark-200 mb-3">Summary</h3>
                <p className="text-dark-400 text-sm">{result.summary}</p>
              </CardContent>
            </Card>

            {/* Model Info Badge */}
            <Card hover={false} className="border-violet-500/20 bg-violet-500/5">
              <CardContent className="py-3">
                <div className="flex items-center gap-2 text-xs text-dark-400">
                  <Cpu className="w-4 h-4 text-violet-400" />
                  <span>Hybrid GNN + GraphCodeBERT</span>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {['CFG Builder', 'GAT Layers', 'Static Features'].map(c => (
                    <span key={c} className="text-xs px-2 py-0.5 bg-dark-700/50 rounded-full text-dark-400">{c}</span>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Right Column */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Tabs */}
            <div className="flex gap-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? tab.id === 'ai'
                        ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                        : 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                      : 'bg-dark-800/50 text-dark-400 hover:text-dark-200'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                  {tab.id === 'ai' && (
                    <span className="text-xs px-1.5 py-0.5 bg-violet-500/20 text-violet-300 rounded-full">
                      New
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              {activeTab === 'overview' && (
                <motion.div
                  key="overview"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  {result.vulnerabilities.map((vuln, index) => (
                    <VulnerabilityCard
                      key={vuln.type}
                      vuln={vuln}
                      index={index}
                      code={code}
                      onAIHelp={(vulnType) => {
                        setAiVuln(vulnType)
                        setActiveTab('ai')
                      }}
                    />
                  ))}

                  {result.recommendations.length > 0 && (
                    <Card hover={false}>
                      <CardContent>
                        <h3 className="text-lg font-semibold text-dark-200 mb-4 flex items-center gap-2">
                          <Zap className="w-5 h-5 text-primary-400" />
                          Recommendations
                        </h3>
                        <ul className="space-y-3">
                          {result.recommendations.map((rec, index) => (
                            <motion.li
                              key={index}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.1 }}
                              className="flex items-start gap-3"
                            >
                              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                              <span className="text-dark-300 text-sm">{rec}</span>
                            </motion.li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}
                </motion.div>
              )}

              {activeTab === 'code' && (
                <motion.div
                  key="code"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Card hover={false}>
                    <CardContent>
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-dark-200">Attention Heatmap</h3>
                        <div className="flex items-center gap-4 text-xs">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-risk-safe" />
                            <span className="text-dark-400">Safe</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-risk-medium" />
                            <span className="text-dark-400">Medium</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-risk-critical" />
                            <span className="text-dark-400">Critical</span>
                          </div>
                        </div>
                      </div>
                      <CodeHeatmap code={code} lineRisks={result.line_risks} />
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {activeTab === 'ai' && (
                <motion.div
                  key="ai"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <AIAssistantPanel
                    code={code}
                    vulnerabilities={result.vulnerabilities}
                    initialVuln={aiVuln}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
