import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload,
  FileCode,
  Play,
  Trash2,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  FileText,
  ChevronDown
} from 'lucide-react'
import { Button, Card, CardContent, Badge, SkeletonCode } from '../components/ui'
import { analyzeContract, getSampleContracts } from '../services/api'

// Default placeholder code
const placeholderCode = `// Paste your Solidity contract here or select a sample contract

pragma solidity ^0.8.0;

contract Example {
    // Your contract code here
}
`

const fallbackSamples = [
  {
    id: 'sample-reentrancy',
    name: 'Simple Reentrancy',
    description: 'Unsafe withdraw before state update.',
    expected_vulnerabilities: ['Reentrancy'],
    code: `pragma solidity ^0.8.0;

contract Vault {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");
        balances[msg.sender] -= amount;
    }
}
`,
  },
  {
    id: 'sample-access-control',
    name: 'Access Control',
    description: 'tx.origin used for authorization.',
    expected_vulnerabilities: ['Access Control'],
    code: `pragma solidity ^0.8.0;

contract AuthExample {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function privileged() external {
        require(tx.origin == owner, "Not owner");
    }
}
`,
  },
]

export default function Analyze() {
  const [code, setCode] = useState('')
  const [sampleContracts, setSampleContracts] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [result, setResult] = useState(null)
  const editorRef = useRef(null)
  const highlightRef = useRef(null)

  // Fetch sample contracts
  useEffect(() => {
    getSampleContracts()
      .then((data) => setSampleContracts(data.contracts))
      .catch((err) => {
        console.error('Failed to load samples:', err)
        setSampleContracts(fallbackSamples)
      })
  }, [])

  // Handle sample selection
  const handleSelectSample = (sampleId) => {
    const sample = sampleContracts.find((item) => item.id === sampleId)
    if (!sample) return
    setSelectedSample(sample)
    setCode(sample.code)
    setError(null)
  }

  // Handle file upload
  const handleFileUpload = useCallback((file) => {
    if (file && file.name.endsWith('.sol')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setCode(e.target.result)
        setSelectedSample(null)
        setError(null)
      }
      reader.readAsText(file)
    } else {
      setError('Please upload a .sol file')
    }
  }, [])

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFileUpload(file)
  }

  // Copy to clipboard
  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Clear code
  const handleClear = () => {
    setCode('')
    setSelectedSample(null)
    setError(null)
  }

  // Analyze contract
  const handleAnalyze = async () => {
    if (!code.trim()) {
      setError('Please enter some code to analyze')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await analyzeContract(code)
      setResult(response)
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  // Line numbers
  const lineCount = code.split('\n').length || 1

  const highlightedLines = useMemo(() => {
    const text = code || ''
    const lines = text.split('\n')
    const tokenRegex = /(\/\/.*$)|("(?:\\.|[^"])*"|'(?:\\.|[^'])*')|(\b(?:pragma|solidity|contract|function|return|returns|if|else|for|while|require|revert|emit|event|modifier|new)\b)|(\b(?:uint|uint256|int|address|bool|string|mapping|memory|storage|calldata|public|private|external|internal|view|pure|payable)\b)|(\b\d+(\.\d+)?\b)/g

    return lines.map((line, lineIndex) => {
      const parts = []
      let lastIndex = 0
      let match

      while ((match = tokenRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          parts.push({ text: line.slice(lastIndex, match.index), className: 'text-dark-900' })
        }
        if (match[1]) parts.push({ text: match[1], className: 'text-emerald-600' }) // comment
        else if (match[2]) parts.push({ text: match[2], className: 'text-amber-600' }) // string
        else if (match[3]) parts.push({ text: match[3], className: 'text-purple-600' }) // keyword
        else if (match[4]) parts.push({ text: match[4], className: 'text-blue-600' }) // type
        else if (match[5]) parts.push({ text: match[5], className: 'text-orange-600' }) // number
        lastIndex = match.index + match[0].length
      }

      if (lastIndex < line.length) {
        parts.push({ text: line.slice(lastIndex), className: 'text-dark-900' })
      }

      return (
        <div key={`${lineIndex}-${line}`} className="whitespace-pre">
          {parts.length
            ? parts.map((part, index) => (
              <span key={`${lineIndex}-${index}`} className={part.className}>
                {part.text}
              </span>
            ))
            : <span className="text-dark-900">{' '}</span>}
        </div>
      )
    })
  }, [code])

  return (
    <div className="pt-24 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <Badge variant="primary" className="mb-4">
            <FileCode className="w-4 h-4 mr-2" />
            Contract Analysis
          </Badge>
          <h1 className="text-4xl font-bold text-dark-900 mb-4">
            Analyze Your Smart Contract
          </h1>
          <p className="text-dark-700 max-w-2xl mx-auto">
            Paste your Solidity code below or upload a .sol file to detect
            vulnerabilities with our AI-powered analysis engine.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1 space-y-4"
          >
            {/* Sample Contracts Dropdown */}
            <Card hover={false}>
              <CardContent>
                <h3 className="text-sm font-semibold text-dark-800 mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Sample Contracts
                </h3>
                <div className="relative">
                  <select
                    value={selectedSample ? selectedSample.id : ''}
                    onChange={(e) => handleSelectSample(e.target.value)}
                    className="w-full appearance-none px-4 py-2.5 rounded-lg bg-white/90 border border-dark-200 text-dark-800 focus:outline-none focus:ring-2 focus:ring-primary-200"
                  >
                    <option value="" disabled>
                      {sampleContracts.length ? 'Select a sample...' : 'Samples unavailable'}
                    </option>
                    {sampleContracts.map((sample) => (
                      <option key={sample.id} value={sample.id}>
                        {sample.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
                </div>
              </CardContent>
            </Card>

            {/* File Upload */}
            <Card hover={false}>
              <CardContent>
                <h3 className="text-sm font-semibold text-dark-200 mb-3 flex items-center gap-2">
                  <Upload className="w-4 h-4" />
                  Upload File
                </h3>
                <label
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`block w-full p-6 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${isDragging
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-dark-600 hover:border-dark-500 hover:bg-dark-800/30'
                    }`}
                >
                  <input
                    type="file"
                    accept=".sol"
                    onChange={(e) => handleFileUpload(e.target.files[0])}
                    className="hidden"
                  />
                  <Upload className={`w-8 h-8 mx-auto mb-2 ${isDragging ? 'text-primary-400' : 'text-dark-400'}`} />
                  <p className="text-sm text-dark-400">
                    {isDragging ? 'Drop file here' : 'Click or drag .sol file'}
                  </p>
                </label>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card hover={false}>
              <CardContent>
                <h3 className="text-sm font-semibold text-dark-200 mb-3">
                  Quick Actions
                </h3>
                <div className="space-y-2">
                  <button
                    onClick={handleCopy}
                    disabled={!code}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-800/50 hover:bg-dark-700/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4 text-dark-400" />
                    )}
                    <span className="text-sm text-dark-300">
                      {copied ? 'Copied!' : 'Copy Code'}
                    </span>
                  </button>
                  <button
                    onClick={handleClear}
                    disabled={!code}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-800/50 hover:bg-dark-700/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Trash2 className="w-4 h-4 text-dark-400" />
                    <span className="text-sm text-dark-300">Clear</span>
                  </button>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Main Editor Area */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-3"
          >
            <Card hover={false} className="overflow-hidden border border-dark-100/70 bg-white">
              <div className="flex items-center justify-between px-4 py-3 border-b border-dark-100/70 bg-white/90">
                <span className="text-sm font-semibold text-dark-800">
                  {selectedSample ? `${selectedSample.id}.sol` : 'contract.sol'}
                </span>
                <div className="text-sm text-dark-500">{lineCount} lines</div>
              </div>

              {/* Code Editor */}
              <div className="relative">
                <div className="flex">
                  {/* Line Numbers */}
                  <div className="flex-shrink-0 py-4 px-2 bg-dark-50 text-right select-none border-r border-dark-100/70">
                    {Array.from({ length: Math.max(lineCount, 20) }).map((_, i) => (
                      <div key={i} className="text-xs text-dark-500 leading-6 px-2">
                        {i + 1}
                      </div>
                    ))}
                  </div>

                  {/* Highlighted Layer */}
                  <div className="relative flex-1 min-h-[500px] bg-white">
                    <pre
                      ref={highlightRef}
                      aria-hidden="true"
                      className="absolute inset-0 m-0 p-4 font-mono text-sm leading-6 whitespace-pre-wrap overflow-auto"
                    >
                      {highlightedLines}
                    </pre>
                    <textarea
                      ref={editorRef}
                      value={code}
                      onChange={(e) => {
                        setCode(e.target.value)
                        setSelectedSample(null)
                        setError(null)
                      }}
                      onScroll={(e) => {
                        if (highlightRef.current) {
                          highlightRef.current.scrollTop = e.target.scrollTop
                          highlightRef.current.scrollLeft = e.target.scrollLeft
                        }
                      }}
                      placeholder={placeholderCode}
                      className="relative z-10 w-full min-h-[500px] p-4 bg-transparent text-transparent caret-dark-900 font-mono text-sm leading-6 resize-none focus:outline-none placeholder:text-dark-300"
                      spellCheck={false}
                    />
                  </div>
                </div>
              </div>

              {/* Editor Footer */}
              <div className="px-4 py-4 border-t border-dark-100/70 bg-white/90">
                {/* Error Message */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="flex items-center gap-2 px-4 py-2 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400"
                    >
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      <span className="text-sm">{error}</span>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Analyze Button */}
                <div className="flex items-center justify-between">
                  <div className="text-sm text-dark-500">
                    {code.length.toLocaleString()} characters
                  </div>
                  <Button
                    onClick={handleAnalyze}
                    disabled={!code.trim() || isLoading}
                    loading={isLoading}
                    icon={isLoading ? null : Play}
                    size="lg"
                  >
                    {isLoading ? 'Analyzing...' : 'Analyze Contract'}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Info Cards */}
            <div className="grid md:grid-cols-3 gap-4 mt-6">
              {[
                {
                  icon: '🔍',
                  title: 'Deep Analysis',
                  description: 'Our AI examines every line of your code'
                },
                {
                  icon: '⚡',
                  title: 'Instant Results',
                  description: 'Get vulnerability reports in seconds'
                },
                {
                  icon: '🎯',
                  title: 'Precise Detection',
                  description: '4 critical vulnerability classes'
                }
              ].map((item, index) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="glass-card p-4"
                >
                  <div className="text-2xl mb-2">{item.icon}</div>
                  <h4 className="font-semibold text-dark-800 mb-1">{item.title}</h4>
                  <p className="text-sm text-dark-700">{item.description}</p>
                </motion.div>
              ))}
            </div>

            {/* Inline Results */}
            {result && (
              <div id="analysis-results" className="mt-8">
                <Card hover={false} className="overflow-hidden">
                  <CardContent>
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                      <div>
                        <h3 className="text-lg font-semibold text-dark-900">Analysis Summary</h3>
                        <p className="text-sm text-dark-600 mt-1">
                          {result.summary || 'Backend analysis completed successfully.'}
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge variant="primary">
                          {result.risk_level || 'Medium'} Risk
                        </Badge>
                        <span className="text-sm text-dark-600">
                          Score: {Math.round((result.overall_risk_score || 0) * 100)}%
                        </span>
                      </div>
                    </div>

                    <div className="mt-6 grid gap-4 md:grid-cols-2">
                      {(result.vulnerabilities || []).map((vuln) => (
                        <div key={vuln.type} className="rounded-xl border border-dark-200/40 bg-white/70 px-4 py-3">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-dark-900">{vuln.type}</div>
                              <div className="text-xs text-dark-600 mt-1">
                                {vuln.description || 'Detected by the backend model.'}
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-semibold text-dark-900">
                                {Math.round((vuln.probability || 0) * 100)}%
                              </div>
                              <div className="text-[11px] text-dark-500">Confidence</div>
                            </div>
                          </div>
                          <div className="mt-3 h-2 rounded-full bg-dark-200/40 overflow-hidden">
                            <div
                              className="h-full bg-primary-500"
                              style={{ width: `${Math.round((vuln.probability || 0) * 100)}%` }}
                            />
                          </div>
                          {vuln.affected_lines && vuln.affected_lines.length > 0 && (
                            <div className="mt-2 text-[11px] text-dark-500">
                              Lines: {vuln.affected_lines.join(', ')}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}
