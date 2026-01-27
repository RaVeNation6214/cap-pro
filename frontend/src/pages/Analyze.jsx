import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
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

export default function Analyze() {
  const navigate = useNavigate()
  const [code, setCode] = useState('')
  const [sampleContracts, setSampleContracts] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  // Fetch sample contracts
  useEffect(() => {
    getSampleContracts()
      .then((data) => setSampleContracts(data.contracts))
      .catch((err) => console.error('Failed to load samples:', err))
  }, [])

  // Handle sample selection
  const handleSelectSample = (sample) => {
    setSelectedSample(sample)
    setCode(sample.code)
    setIsDropdownOpen(false)
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
      const result = await analyzeContract(code)
      // Store result and navigate to results page
      sessionStorage.setItem('analysisResult', JSON.stringify(result))
      sessionStorage.setItem('analyzedCode', code)
      navigate('/results')
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  // Line numbers
  const lineCount = code.split('\n').length || 1

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
          <h1 className="text-4xl font-bold text-dark-100 mb-4">
            Analyze Your Smart Contract
          </h1>
          <p className="text-dark-400 max-w-2xl mx-auto">
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
                <h3 className="text-sm font-semibold text-dark-200 mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Sample Contracts
                </h3>
                <div className="relative">
                  <button
                    onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    className="w-full px-4 py-2.5 rounded-lg bg-dark-800/50 border border-dark-600 text-left flex items-center justify-between hover:border-dark-500 transition-colors"
                  >
                    <span className="text-dark-300 truncate">
                      {selectedSample ? selectedSample.name : 'Select a sample...'}
                    </span>
                    <ChevronDown className={`w-4 h-4 text-dark-400 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
                  </button>

                  <AnimatePresence>
                    {isDropdownOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full left-0 right-0 mt-2 z-20 glass rounded-lg border border-dark-600 overflow-hidden"
                      >
                        {sampleContracts.map((sample) => (
                          <button
                            key={sample.id}
                            onClick={() => handleSelectSample(sample)}
                            className="w-full px-4 py-3 text-left hover:bg-dark-700/50 transition-colors border-b border-dark-700/50 last:border-0"
                          >
                            <div className="font-medium text-dark-200 text-sm">
                              {sample.name}
                            </div>
                            <div className="text-xs text-dark-400 mt-0.5">
                              {sample.description}
                            </div>
                            <div className="flex gap-1 mt-2">
                              {sample.expected_vulnerabilities.map((vuln) => (
                                <Badge key={vuln} variant="danger" size="sm">
                                  {vuln}
                                </Badge>
                              ))}
                            </div>
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
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
                  className={`block w-full p-6 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${
                    isDragging
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
            <Card hover={false} className="overflow-hidden">
              {/* Editor Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-dark-700/50 bg-dark-800/30">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500/70" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                    <div className="w-3 h-3 rounded-full bg-green-500/70" />
                  </div>
                  <span className="text-sm text-dark-400 font-mono">
                    {selectedSample ? `${selectedSample.id}.sol` : 'contract.sol'}
                  </span>
                </div>
                <div className="text-sm text-dark-500">
                  {lineCount} lines
                </div>
              </div>

              {/* Code Editor */}
              <div className="relative">
                <div className="flex">
                  {/* Line Numbers */}
                  <div className="flex-shrink-0 py-4 px-2 bg-dark-900/50 text-right select-none border-r border-dark-700/50">
                    {Array.from({ length: Math.max(lineCount, 20) }).map((_, i) => (
                      <div key={i} className="text-xs text-dark-500 leading-6 px-2">
                        {i + 1}
                      </div>
                    ))}
                  </div>

                  {/* Textarea */}
                  <textarea
                    value={code}
                    onChange={(e) => {
                      setCode(e.target.value)
                      setSelectedSample(null)
                      setError(null)
                    }}
                    placeholder={placeholderCode}
                    className="flex-1 min-h-[500px] p-4 bg-transparent text-dark-100 font-mono text-sm leading-6 resize-none focus:outline-none placeholder-dark-600"
                    spellCheck={false}
                  />
                </div>
              </div>

              {/* Editor Footer */}
              <div className="px-4 py-4 border-t border-dark-700/50 bg-dark-800/30">
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
                  <h4 className="font-semibold text-dark-200 mb-1">{item.title}</h4>
                  <p className="text-sm text-dark-400">{item.description}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
