import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
<<<<<<< HEAD
import { useNavigate } from 'react-router-dom'
=======
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload,
  FileCode,
  Play,
  Trash2,
  Copy,
  Check,
  AlertCircle,
  FileText,
  ChevronDown
} from 'lucide-react'
import { Button, Card, CardContent, Badge, SkeletonCode } from '../components/ui'
import { analyzeContract, getSampleContracts } from '../services/api'

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
<<<<<<< HEAD
=======
  const [result, setResult] = useState(null)
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
  const editorRef = useRef(null)
  const highlightRef = useRef(null)

  useEffect(() => {
    getSampleContracts()
      .then((data) => setSampleContracts(data.contracts))
<<<<<<< HEAD
      .catch(() => setSampleContracts(fallbackSamples))
  }, [])

=======
      .catch((err) => {
        console.error('Failed to load samples:', err)
        setSampleContracts(fallbackSamples)
      })
  }, [])

  // Handle sample selection
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
  const handleSelectSample = (sampleId) => {
    const sample = sampleContracts.find((item) => item.id === sampleId)
    if (!sample) return
    setSelectedSample(sample)
    setCode(sample.code)
    setError(null)
  }

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

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false) }
  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFileUpload(e.dataTransfer.files[0])
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleClear = () => {
    setCode('')
    setSelectedSample(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!code.trim()) {
      setError('Please enter some code to analyze')
      return
    }
    setIsLoading(true)
    setError(null)
    try {
<<<<<<< HEAD
      const result = await analyzeContract(code)
      sessionStorage.setItem('analysisResult', JSON.stringify(result))
      sessionStorage.setItem('analyzedCode', code)
      navigate('/results')
=======
      const response = await analyzeContract(code)
      setResult(response)
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const lineCount = code.split('\n').length || 1

  const highlightedLines = useMemo(() => {
    const text = code || ''
    const lines = text.split('\n')
<<<<<<< HEAD
    const tokenRegex = /(\/\/.*$)|("(?:\.|[^"])*"|'(?:\.|[^'])*')|(\b(?:pragma|solidity|contract|function|return|returns|if|else|for|while|require|revert|emit|event|modifier|new)\b)|(\b(?:uint|uint256|int|address|bool|string|mapping|memory|storage|calldata|public|private|external|internal|view|pure|payable)\b)|(\b\d+(\.\d+)?\b)/g
=======
    const tokenRegex = /(\/\/.*$)|("(?:\\.|[^"])*"|'(?:\\.|[^'])*')|(\b(?:pragma|solidity|contract|function|return|returns|if|else|for|while|require|revert|emit|event|modifier|new)\b)|(\b(?:uint|uint256|int|address|bool|string|mapping|memory|storage|calldata|public|private|external|internal|view|pure|payable)\b)|(\b\d+(\.\d+)?\b)/g
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934

    return lines.map((line, lineIndex) => {
      const parts = []
      let lastIndex = 0
      let match
<<<<<<< HEAD
      const regex = new RegExp(tokenRegex.source, tokenRegex.flags)

      while ((match = regex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          parts.push({ text: line.slice(lastIndex, match.index), className: 'text-dark-900' })
        }
        if (match[1]) parts.push({ text: match[1], className: 'text-emerald-600' })
        else if (match[2]) parts.push({ text: match[2], className: 'text-amber-600' })
        else if (match[3]) parts.push({ text: match[3], className: 'text-purple-600' })
        else if (match[4]) parts.push({ text: match[4], className: 'text-blue-600' })
        else if (match[5]) parts.push({ text: match[5], className: 'text-orange-600' })
=======

      while ((match = tokenRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          parts.push({ text: line.slice(lastIndex, match.index), className: 'text-dark-900' })
        }
        if (match[1]) parts.push({ text: match[1], className: 'text-emerald-600' }) // comment
        else if (match[2]) parts.push({ text: match[2], className: 'text-amber-600' }) // string
        else if (match[3]) parts.push({ text: match[3], className: 'text-purple-600' }) // keyword
        else if (match[4]) parts.push({ text: match[4], className: 'text-blue-600' }) // type
        else if (match[5]) parts.push({ text: match[5], className: 'text-orange-600' }) // number
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
        lastIndex = match.index + match[0].length
      }

      if (lastIndex < line.length) {
        parts.push({ text: line.slice(lastIndex), className: 'text-dark-900' })
      }

      return (
<<<<<<< HEAD
        <div key={lineIndex + '-' + line.slice(0, 10)} className="whitespace-pre">
          {parts.length
            ? parts.map((part, index) => (
              <span key={lineIndex + '-' + index} className={part.className}>{part.text}</span>
=======
        <div key={`${lineIndex}-${line}`} className="whitespace-pre">
          {parts.length
            ? parts.map((part, index) => (
              <span key={`${lineIndex}-${index}`} className={part.className}>
                {part.text}
              </span>
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
            ))
            : <span className="text-dark-900">{' '}</span>}
        </div>
      )
    })
  }, [code])

  return (
    <div className="pt-24 pb-12 min-h-screen bg-gradient-to-b from-[#f7f6ff] to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
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
<<<<<<< HEAD
          <p className="text-dark-600 max-w-2xl mx-auto">
=======
          <p className="text-dark-700 max-w-2xl mx-auto">
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
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
            <Card hover={false} className="bg-white/80 border-white/70">
              <CardContent>
                <h3 className="text-sm font-semibold text-dark-800 mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Sample Contracts
                </h3>
                <div className="relative">
                  <select
                    value={selectedSample ? selectedSample.id : ''}
                    onChange={(e) => handleSelectSample(e.target.value)}
<<<<<<< HEAD
                    className="w-full appearance-none px-4 py-2.5 rounded-lg bg-white border border-dark-200 text-dark-800 focus:outline-none focus:ring-2 focus:ring-primary-200 text-sm"
                  >
                    <option value="" disabled>
                      {sampleContracts.length ? 'Select a sample...' : 'Loading...'}
=======
                    className="w-full appearance-none px-4 py-2.5 rounded-lg bg-white/90 border border-dark-200 text-dark-800 focus:outline-none focus:ring-2 focus:ring-primary-200"
                  >
                    <option value="" disabled>
                      {sampleContracts.length ? 'Select a sample...' : 'Samples unavailable'}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                    </option>
                    {sampleContracts.map((sample) => (
                      <option key={sample.id} value={sample.id}>
                        {sample.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
                </div>
                {selectedSample && (
                  <p className="mt-2 text-xs text-dark-500">{selectedSample.description}</p>
                )}
              </CardContent>
            </Card>

            <Card hover={false} className="bg-white/80 border-white/70">
              <CardContent>
                <h3 className="text-sm font-semibold text-dark-800 mb-3 flex items-center gap-2">
                  <Upload className="w-4 h-4" />
                  Upload File
                </h3>
                <label
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
<<<<<<< HEAD
                  className={`block w-full p-6 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-dark-300 hover:border-primary-400 hover:bg-primary-50/50'
                  }`}
=======
                  className={`block w-full p-6 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${isDragging
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-dark-600 hover:border-dark-500 hover:bg-dark-800/30'
                    }`}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                >
                  <input
                    type="file"
                    accept=".sol"
                    onChange={(e) => handleFileUpload(e.target.files[0])}
                    className="hidden"
                  />
                  <Upload className={`w-8 h-8 mx-auto mb-2 ${isDragging ? 'text-primary-500' : 'text-dark-400'}`} />
                  <p className="text-sm text-dark-500">
                    {isDragging ? 'Drop file here' : 'Click or drag .sol file'}
                  </p>
                </label>
              </CardContent>
            </Card>

            <Card hover={false} className="bg-white/80 border-white/70">
              <CardContent>
                <h3 className="text-sm font-semibold text-dark-800 mb-3">Quick Actions</h3>
                <div className="space-y-2">
                  <button
                    onClick={handleCopy}
                    disabled={!code}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-50 hover:bg-dark-100 border border-dark-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4 text-dark-500" />
                    )}
                    <span className="text-sm text-dark-700">{copied ? 'Copied!' : 'Copy Code'}</span>
                  </button>
                  <button
                    onClick={handleClear}
                    disabled={!code}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-50 hover:bg-dark-100 border border-dark-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Trash2 className="w-4 h-4 text-dark-500" />
                    <span className="text-sm text-dark-700">Clear</span>
                  </button>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Editor */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-3"
          >
<<<<<<< HEAD
            <div className="rounded-2xl overflow-hidden border border-dark-100/70 bg-white shadow-xl shadow-purple-900/5">
              <div className="flex items-center justify-between px-4 py-3 border-b border-dark-100/70 bg-white/90">
                <span className="text-sm font-semibold text-dark-800">
                  {selectedSample ? selectedSample.id + '.sol' : 'contract.sol'}
=======
            <Card hover={false} className="overflow-hidden border border-dark-100/70 bg-white">
              <div className="flex items-center justify-between px-4 py-3 border-b border-dark-100/70 bg-white/90">
                <span className="text-sm font-semibold text-dark-800">
                  {selectedSample ? `${selectedSample.id}.sol` : 'contract.sol'}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                </span>
                <div className="text-sm text-dark-500">{lineCount} lines</div>
              </div>

              <div className="relative">
                <div className="flex">
<<<<<<< HEAD
=======
                  {/* Line Numbers */}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                  <div className="flex-shrink-0 py-4 px-2 bg-dark-50 text-right select-none border-r border-dark-100/70">
                    {Array.from({ length: Math.max(lineCount, 20) }).map((_, i) => (
                      <div key={i} className="text-xs text-dark-400 leading-6 px-2">{i + 1}</div>
                    ))}
                  </div>
<<<<<<< HEAD
=======

                  {/* Highlighted Layer */}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                  <div className="relative flex-1 min-h-[500px] bg-white">
                    <pre
                      ref={highlightRef}
                      aria-hidden="true"
<<<<<<< HEAD
                      className="absolute inset-0 m-0 p-4 font-mono text-sm leading-6 whitespace-pre-wrap overflow-auto pointer-events-none"
=======
                      className="absolute inset-0 m-0 p-4 font-mono text-sm leading-6 whitespace-pre-wrap overflow-auto"
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                    >
                      {highlightedLines}
                    </pre>
                    <textarea
                      ref={editorRef}
                      value={code}
<<<<<<< HEAD
                      onChange={(e) => { setCode(e.target.value); setSelectedSample(null); setError(null) }}
=======
                      onChange={(e) => {
                        setCode(e.target.value)
                        setSelectedSample(null)
                        setError(null)
                      }}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
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

<<<<<<< HEAD
              <div className="px-4 py-4 border-t border-dark-100/70 bg-white/90">
=======
              {/* Editor Footer */}
              <div className="px-4 py-4 border-t border-dark-100/70 bg-white/90">
                {/* Error Message */}
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="flex items-center gap-2 px-4 py-2 mb-4 rounded-lg bg-red-50 border border-red-200 text-red-600"
                    >
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      <span className="text-sm">{error}</span>
                    </motion.div>
                  )}
                </AnimatePresence>
                <div className="flex items-center justify-between">
                  <div className="text-sm text-dark-500">{code.length.toLocaleString()} characters</div>
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
            </div>

            <div className="grid md:grid-cols-3 gap-4 mt-6">
              {[
                { icon: '🔍', title: 'Deep Analysis', description: 'AI examines every line of your Solidity code' },
                { icon: '⚡', title: 'Instant Results', description: 'Get full vulnerability report in seconds' },
                { icon: '🤖', title: 'AI Assistant', description: 'Gemini-powered explanations and fix suggestions' }
              ].map((item, index) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="rounded-2xl bg-white/70 border border-white/70 shadow-lg shadow-purple-900/5 p-4"
                >
                  <div className="text-2xl mb-2">{item.icon}</div>
                  <h4 className="font-semibold text-dark-800 mb-1">{item.title}</h4>
<<<<<<< HEAD
                  <p className="text-sm text-dark-600">{item.description}</p>
=======
                  <p className="text-sm text-dark-700">{item.description}</p>
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
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
