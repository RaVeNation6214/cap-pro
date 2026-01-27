import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield,
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
  Zap
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

// Animated risk gauge
function RiskGauge({ score, level }) {
  const config = riskConfig[level] || riskConfig.Medium
  const Icon = config.icon
  const rotation = score * 180 // 0 to 180 degrees

  return (
    <div className="relative w-64 h-32 mx-auto">
      {/* Gauge background */}
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

        {/* Background arc */}
        <path
          d="M 20 90 A 80 80 0 0 1 180 90"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* Colored arc */}
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

        {/* Needle */}
        <motion.g
          initial={{ rotate: 0 }}
          animate={{ rotate: rotation }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
          style={{ transformOrigin: '100px 90px' }}
        >
          <line
            x1="100"
            y1="90"
            x2="100"
            y2="30"
            stroke="white"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="100" cy="90" r="8" fill="white" />
        </motion.g>
      </svg>

      {/* Score display */}
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
function VulnerabilityCard({ vuln, index }) {
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
    >
      <Card hover={false} className={`border-l-4 border-l-${variant === 'safe' ? 'green' : variant === 'low' ? 'lime' : variant === 'medium' ? 'yellow' : variant === 'high' ? 'orange' : 'red'}-500`}>
        <CardContent>
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center bg-${variant === 'safe' ? 'green' : variant === 'critical' ? 'red' : variant === 'high' ? 'orange' : variant === 'medium' ? 'yellow' : 'lime'}-500/20`}>
                <Icon className={`w-5 h-5 text-${variant === 'safe' ? 'green' : variant === 'critical' ? 'red' : variant === 'high' ? 'orange' : variant === 'medium' ? 'yellow' : 'lime'}-400`} />
              </div>
              <div>
                <h4 className="font-semibold text-dark-100">{vuln.type}</h4>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant={variant} size="sm">
                    {vuln.confidence} Confidence
                  </Badge>
                  {vuln.affected_lines.length > 0 && (
                    <span className="text-xs text-dark-400">
                      Lines: {vuln.affected_lines.join(', ')}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-2xl font-bold text-dark-100">
                  {(vuln.probability * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-dark-400">Probability</div>
              </div>
              {isExpanded ? (
                <ChevronUp className="w-5 h-5 text-dark-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-dark-400" />
              )}
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
                  <p className="text-dark-400 text-sm mb-4">
                    {vuln.description}
                  </p>
                  <Progress
                    value={vuln.probability * 100}
                    variant={variant}
                    showLabel
                    label="Detection Confidence"
                  />
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
            <code className="code-line-content">
              {line || ' '}
            </code>
            {lineRisk.is_vulnerable && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="flex-shrink-0"
              >
                <AlertTriangle className="w-4 h-4 text-red-400" />
              </motion.div>
            )}
          </motion.div>
        )
      })}
    </div>
  )
}

export default function Results() {
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [code, setCode] = useState('')
  const [activeTab, setActiveTab] = useState('overview')

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

  const riskLevel = riskConfig[result.risk_level] || riskConfig.Medium

  // Prepare chart data
  const chartData = result.vulnerabilities.map((v) => ({
    name: v.type.split(' ')[0],
    probability: v.probability * 100,
    fill: v.probability >= 0.6 ? '#ef4444' : v.probability >= 0.4 ? '#f97316' : v.probability >= 0.2 ? '#eab308' : '#22c55e'
  }))

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
              <Button variant="ghost" icon={ArrowLeft}>
                Back
              </Button>
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-dark-100">Analysis Results</h1>
              <p className="text-dark-400 mt-1">
                Vulnerability detection complete
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" icon={RefreshCw} onClick={() => navigate('/analyze')}>
              New Analysis
            </Button>
            <Button variant="secondary" icon={Download}>
              Export
            </Button>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - Risk Overview */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1 space-y-6"
          >
            {/* Risk Score Card */}
            <Card hover={false}>
              <CardContent className="text-center py-8">
                <h3 className="text-lg font-semibold text-dark-200 mb-6">
                  Overall Risk Score
                </h3>
                <RiskGauge score={result.overall_risk_score} level={result.risk_level} />
              </CardContent>
            </Card>

            {/* Probability Chart */}
            <Card hover={false}>
              <CardContent>
                <h3 className="text-lg font-semibold text-dark-200 mb-4">
                  Vulnerability Probabilities
                </h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical">
                      <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 12 }} />
                      <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={80} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: '8px',
                        }}
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
                <h3 className="text-lg font-semibold text-dark-200 mb-3">
                  Summary
                </h3>
                <p className="text-dark-400 text-sm">
                  {result.summary}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Right Column - Details */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Tabs */}
            <div className="flex gap-2">
              {[
                { id: 'overview', label: 'Overview', icon: Eye },
                { id: 'code', label: 'Code Analysis', icon: Code },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                      : 'bg-dark-800/50 text-dark-400 hover:text-dark-200'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              {activeTab === 'overview' ? (
                <motion.div
                  key="overview"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  {/* Vulnerability Cards */}
                  {result.vulnerabilities.map((vuln, index) => (
                    <VulnerabilityCard key={vuln.type} vuln={vuln} index={index} />
                  ))}

                  {/* Recommendations */}
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
              ) : (
                <motion.div
                  key="code"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Card hover={false}>
                    <CardContent>
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-dark-200">
                          Attention Heatmap
                        </h3>
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
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
