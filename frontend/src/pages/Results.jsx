import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle, CheckCircle, XCircle, ArrowLeft, Download, Share2,
  RefreshCw, Info, Bug, Activity, Send, Copy, Check, Target, Layers,
  Network, Table2, TrendingUp, Eye, Code2, GitBranch, Bot, Sparkles,
  FileText, Shield, ChevronRight, BarChart3
} from 'lucide-react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip,
  LineChart, Line, CartesianGrid,
  PieChart, Pie, RadialBarChart, RadialBar, Treemap
} from 'recharts'
import {
  ReactFlow, Background, Controls, MiniMap,
  useNodesState, useEdgesState, MarkerType, Handle, Position,
  getBezierPath
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css'
import { sendChat, runSlither, runMythril } from '../services/api'

// ── Constants ──────────────────────────────────────────────────────────────
const RISK_CFG = {
  Safe:     { hex: '#16a34a', bg: 'bg-green-50',  border: 'border-green-200',  text: 'text-green-700',  label: 'SAFE',     icon: CheckCircle },
  Low:      { hex: '#65a30d', bg: 'bg-lime-50',   border: 'border-lime-200',   text: 'text-lime-700',   label: 'LOW',      icon: Info },
  Medium:   { hex: '#ca8a04', bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', label: 'MEDIUM',   icon: AlertTriangle },
  High:     { hex: '#ea580c', bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700', label: 'HIGH',     icon: AlertTriangle },
  Critical: { hex: '#dc2626', bg: 'bg-red-50',    border: 'border-red-200',    text: 'text-red-700',    label: 'CRITICAL', icon: XCircle },
}

const VULN_COLORS = {
  Reentrancy:        '#ef4444',
  Arithmetic:        '#f97316',
  'Access Control':  '#8b5cf6',
  'Unchecked Calls': '#3b82f6',
  Timestamp:         '#eab308',
}

const SWC_MAP = {
  Reentrancy: 'SWC-107', Arithmetic: 'SWC-101',
  'Access Control': 'SWC-115', 'Unchecked Calls': 'SWC-104', Timestamp: 'SWC-116',
}

// Node type colors (white theme)
const NODE_STYLES = {
  entry:         { bg: '#dcfce7', border: '#16a34a', text: '#14532d', icon: '▶' },
  exit:          { bg: '#f1f5f9', border: '#94a3b8', text: '#475569', icon: '■' },
  function:      { bg: '#eff6ff', border: '#3b82f6', text: '#1e40af', icon: 'ƒ' },
  block:         { bg: '#f8fafc', border: '#cbd5e1', text: '#475569', icon: '≡' },
  branch:        { bg: '#fefce8', border: '#eab308', text: '#713f12', icon: '⟐' },
  loop:          { bg: '#f0f9ff', border: '#0ea5e9', text: '#0c4a6e', icon: '↺' },
  external_call: { bg: '#fef2f2', border: '#ef4444', text: '#7f1d1d', icon: '⚠' },
  access_control:{ bg: '#fdf4ff', border: '#a855f7', text: '#581c87', icon: '🔒' },
  state_write:   { bg: '#fdf4ff', border: '#8b5cf6', text: '#4c1d95', icon: '✎' },
  guard:         { bg: '#f0fdf4', border: '#22c55e', text: '#14532d', icon: '✓' },
  return:        { bg: '#f8fafc', border: '#64748b', text: '#334155', icon: '←' },
  contract:      { bg: '#eef2ff', border: '#6366f1', text: '#3730a3', icon: '◈' },
}

const EDGE_COLORS = {
  control: '#94a3b8',
  true:    '#22c55e',
  false:   '#ef4444',
  loop:    '#0ea5e9',
  risky:   '#dc2626',
  data:    '#8b5cf6',
}

// ── Helpers ────────────────────────────────────────────────────────────────
const pct = (v) => Math.round((v || 0) * 100)
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

function riskColor(score) {
  if (score >= 0.8) return '#dc2626'
  if (score >= 0.6) return '#ea580c'
  if (score >= 0.4) return '#ca8a04'
  if (score >= 0.2) return '#65a30d'
  return '#16a34a'
}

// ── Custom CFG node ────────────────────────────────────────────────────────
function CFGNode({ data }) {
  const style = NODE_STYLES[data.node_type] || NODE_STYLES.block
  const isRisky = data.risk_score > 0.5
  return (
    <div style={{
      background: style.bg,
      border: `2px solid ${isRisky ? '#ef4444' : style.border}`,
      borderRadius: data.node_type === 'branch' ? 6 : 8,
      padding: '6px 12px',
      minWidth: 110,
      maxWidth: 160,
      boxShadow: isRisky ? '0 0 0 3px rgba(239,68,68,0.15)' : '0 1px 3px rgba(0,0,0,0.1)',
      cursor: 'pointer',
      transform: data.node_type === 'branch' ? 'rotate(0deg)' : 'none',
    }}>
      <Handle type="target" position={Position.Top} style={{ background: style.border, width: 8, height: 8 }} />
      <div style={{ color: style.text, fontWeight: 700, fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
        <span style={{ fontSize: 13 }}>{style.icon}</span>
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{data.label}</span>
      </div>
      {data.risk_score > 0.05 && (
        <div style={{ fontSize: 10, color: riskColor(data.risk_score), fontWeight: 600, marginTop: 2 }}>
          risk: {Math.round(data.risk_score * 100)}%
        </div>
      )}
      {data.vuln_types?.length > 0 && (
        <div style={{ fontSize: 9, color: '#6b7280', marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {data.vuln_types.join(', ')}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ background: style.border, width: 8, height: 8 }} />
    </div>
  )
}

const NODE_TYPES = { cfgNode: CFGNode }

// ── Animated risk ring ─────────────────────────────────────────────────────
function RiskRing({ score, level }) {
  const cfg = RISK_CFG[level] || RISK_CFG.Safe
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    let frame
    const target = pct(score)
    const animate = () => {
      setDisplay(prev => {
        const next = prev + (target - prev) * 0.09
        if (Math.abs(next - target) < 0.5) return target
        frame = requestAnimationFrame(animate)
        return next
      })
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [score])

  return (
    <div style={{ width: 140, height: 140 }}>
      <CircularProgressbar value={display} text={`${Math.round(display)}%`}
        styles={buildStyles({
          textSize: '20px', pathColor: cfg.hex, textColor: cfg.hex, trailColor: '#f1f5f9',
        })} />
    </div>
  )
}

// ── Markdown renderer ─────────────────────────────────────────────────────
function MD({ text }) {
  const html = (text || '')
    .replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre class="bg-gray-50 border border-gray-200 rounded p-3 overflow-x-auto text-xs my-2"><code>$1</code></pre>')
    .replace(/^### (.*)/gm, '<h3 class="text-gray-800 font-bold text-sm mt-3 mb-1">$1</h3>')
    .replace(/^## (.*)/gm, '<h2 class="text-gray-900 font-bold text-base mt-4 mb-2">$1</h2>')
    .replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-900">$1</strong>')
    .replace(/\*(.*?)\*/g, '<em class="text-gray-600">$1</em>')
    .replace(/\n/g, '<br/>')
  return <div dangerouslySetInnerHTML={{ __html: html }} className="text-gray-700 text-xs leading-relaxed" />
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function Results() {
  const navigate = useNavigate()
  const [analysis, setAnalysis]   = useState(null)
  const [code, setCode]           = useState('')
  const [activeTab, setActiveTab] = useState('overview')
  const [copied, setCopied]       = useState(null)
  const [sortCol, setSortCol]     = useState('risk_score')
  const [sortDir, setSortDir]     = useState('desc')

  // Static analysis state
  const [slitherResult, setSlitherResult] = useState(null)
  const [mythrilResult, setMythrilResult] = useState(null)
  const [staticLoading, setStaticLoading] = useState(false)

  // Chat sidebar state
  const [chatOpen, setChatOpen]   = useState(false)
  const [chatMsgs, setChatMsgs]   = useState([
    { role: 'assistant', content: '👋 Hi! I\'m your audit assistant. I have full context of this contract\'s analysis. Ask me about the vulnerabilities found, how to fix them, or anything about smart contract security.' }
  ])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const chatEndRef = useRef(null)

  // React Flow
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState([])
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState(null)

  // ── Load analysis + run static tools ─────────────────────────────────────
  useEffect(() => {
    const stored = sessionStorage.getItem('analysisResult')
    const storedCode = sessionStorage.getItem('contractCode')
    if (!stored) { navigate('/analyze'); return }
    const data = JSON.parse(stored)
    setAnalysis(data)
    const contractCode = storedCode || ''
    setCode(contractCode)
    buildFlow(data)

    // Run Slither + Mythril in parallel on the real contract code
    if (contractCode) {
      setStaticLoading(true)
      Promise.all([
        runSlither(contractCode).catch(e => ({ available: false, findings: [], error: e.message })),
        runMythril(contractCode).catch(e => ({ available: false, findings: [], error: e.message })),
      ]).then(([sl, my]) => {
        setSlitherResult(sl)
        setMythrilResult(my)
      }).finally(() => setStaticLoading(false))
    }
  }, [navigate])

  // ── Build React Flow graph ─────────────────────────────────────────────
  const buildFlow = useCallback((data) => {
    if (!data?.cfg_nodes?.length) return

    // Layout: place nodes in a hierarchical grid
    // Assign columns based on node type for a flow-like appearance
    const typeOrder = ['entry','function','block','branch','loop','external_call','access_control','state_write','guard','return','exit']
    const rows = []
    let currentRow = []
    let rowWidth = 0

    // Simple layered layout: group into rows of max 3
    data.cfg_nodes.forEach((n, i) => {
      currentRow.push(n)
      rowWidth++
      if (rowWidth >= 3 || n.node_type === 'exit' || n.node_type === 'return') {
        rows.push([...currentRow])
        currentRow = []
        rowWidth = 0
      }
    })
    if (currentRow.length) rows.push(currentRow)

    const nodeW = 180, nodeH = 90, gapX = 220, gapY = 110

    const flowNodes = []
    rows.forEach((row, ri) => {
      const totalW = row.length * gapX
      row.forEach((n, ci) => {
        const x = ci * gapX - (row.length - 1) * gapX / 2 + 300
        const y = ri * gapY + 40
        flowNodes.push({
          id: n.id,
          type: 'cfgNode',
          position: { x, y },
          data: { ...n },
        })
      })
    })

    const flowEdges = (data.cfg_edges || []).map((e, i) => {
      const color = EDGE_COLORS[e.edge_type] || '#94a3b8'
      const isLoop = e.edge_type === 'loop'
      const isRisky = e.edge_type === 'risky'
      return {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        animated: isRisky || isLoop,
        style: {
          stroke: color,
          strokeWidth: isRisky ? 2.5 : 1.5,
          strokeDasharray: e.edge_type === 'false' ? '4 3' : undefined,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
        label: e.edge_type === 'true' ? 'T' : e.edge_type === 'false' ? 'F' : e.edge_type === 'loop' ? '↺' : undefined,
        labelStyle: { fontSize: 10, fontWeight: 700, fill: color },
        type: isLoop ? 'smoothstep' : 'default',
      }
    })

    setRfNodes(flowNodes)
    setRfEdges(flowEdges)
  }, [setRfNodes, setRfEdges])

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMsgs])

  // ── Computed ───────────────────────────────────────────────────────────
  const detectedVulns = useMemo(() =>
    (analysis?.vulnerabilities || []).filter(v => v.probability > 0.4), [analysis])

  const analysisCtx = useMemo(() => {
    if (!analysis) return ''
    return `Risk: ${analysis.risk_level} (${pct(analysis.overall_risk_score)}%). Detected: ${detectedVulns.map(v => v.type).join(', ')}.`
  }, [analysis, detectedVulns])

  const sortedFunctions = useMemo(() =>
    [...(analysis?.function_metrics || [])].sort((a, b) => {
      const av = a[sortCol] ?? 0, bv = b[sortCol] ?? 0
      return sortDir === 'asc' ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
    }), [analysis, sortCol, sortDir])

  const lineRiskData = useMemo(() =>
    (analysis?.line_risks || []).filter(l => l.risk_score > 0.1).slice(0, 50).map(l => ({
      name: `L${l.line_number}`, risk: Math.round(l.risk_score * 100), fill: riskColor(l.risk_score),
    })), [analysis])

  const attentionData = useMemo(() =>
    (analysis?.attention_weights || []).slice(0, 30).map((w, i) => ({
      window: i + 1, attention: Math.round(w * 100),
    })), [analysis])

  const pieData = useMemo(() =>
    detectedVulns.map(v => ({ name: v.type, value: Math.round(v.probability * 100), fill: VULN_COLORS[v.type] || '#6b7280' })),
    [detectedVulns])

  const radialData = useMemo(() =>
    (analysis?.vulnerabilities || []).map(v => ({
      name: v.type.split(' ')[0], value: Math.round(v.probability * 100), fill: VULN_COLORS[v.type] || '#6b7280',
    })), [analysis])

  // ── Tri-tool comparison ───────────────────────────────────────────────
  const triComparison = useMemo(() => {
    const VULN_ROWS = [
      { name: 'Reentrancy',       swc: 'SWC-107', slitherKeys: ['reentrancy-eth','reentrancy-no-eth','reentrancy-benign','reentrancy-events'], mythrilSwcs: ['SWC-107'] },
      { name: 'Arithmetic',       swc: 'SWC-101', slitherKeys: ['integer-overflow','tautology','divide-before-multiply'],                       mythrilSwcs: ['SWC-101'] },
      { name: 'Access Control',   swc: 'SWC-115', slitherKeys: ['tx-origin','access-control'],                                                  mythrilSwcs: ['SWC-115','SWC-105','SWC-106'] },
      { name: 'Unchecked Calls',  swc: 'SWC-104', slitherKeys: ['unchecked-lowlevel','unchecked-send','unchecked-transfer','low-level-calls'],   mythrilSwcs: ['SWC-104','SWC-112'] },
      { name: 'Timestamp',        swc: 'SWC-116', slitherKeys: ['timestamp','block-timestamp'],                                                  mythrilSwcs: ['SWC-116','SWC-120'] },
    ]
    const slitherChecks = new Set((slitherResult?.findings || []).map(f => f.check?.toLowerCase()))
    const mythrilSwcs   = new Set((mythrilResult?.findings  || []).map(f => f.swc?.toUpperCase()))

    return VULN_ROWS.map(row => {
      const modelVuln = (analysis?.vulnerabilities || []).find(v => v.type === row.name)
      const modelProb = modelVuln?.probability || 0
      const modelHit  = modelProb > 0.4

      const slitherHit  = row.slitherKeys.some(k => slitherChecks.has(k))
      const mythrilHit  = row.mythrilSwcs.some(s => mythrilSwcs.has(s))

      const hits = [modelHit, slitherHit, mythrilHit].filter(Boolean).length
      const consensus = hits === 3 ? 'all' : hits === 2 ? 'two' : hits === 1 ? 'one' : 'none'

      // Get actual slither/mythril finding details for this vuln
      const slitherFindings = (slitherResult?.findings || []).filter(f => row.slitherKeys.includes(f.check?.toLowerCase()))
      const mythrilFindings = (mythrilResult?.findings  || []).filter(f => row.mythrilSwcs.includes(f.swc?.toUpperCase()))

      return { ...row, modelProb, modelHit, slitherHit, mythrilHit, consensus, slitherFindings, mythrilFindings }
    })
  }, [analysis, slitherResult, mythrilResult])

  // ── Handlers ──────────────────────────────────────────────────────────
  const handleNodeClick = useCallback((_, node) => {
    setSelectedNode(node.data)
    setActiveTab('code')
  }, [])

  function sortBy(col) {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('desc') }
  }

  async function sendChatMsg() {
    if (!chatInput.trim() || chatLoading) return
    const userMsg = { role: 'user', content: chatInput.trim() }
    const history = [...chatMsgs, userMsg]
    setChatMsgs(history)
    setChatInput('')
    setChatLoading(true)
    try {
      const res = await sendChat(history.filter(m => m.role !== 'system'), code, analysisCtx)
      setChatMsgs(prev => [...prev, { role: 'assistant', content: res.reply }])
    } catch {
      setChatMsgs(prev => [...prev, { role: 'assistant', content: '❌ AI unavailable. Set GEMINI_API_KEY in backend/.env' }])
    } finally { setChatLoading(false) }
  }

  function exportJSON() {
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'audit-report.json'; a.click()
    URL.revokeObjectURL(url)
  }

  function copyText(text, id) {
    navigator.clipboard.writeText(text); setCopied(id); setTimeout(() => setCopied(null), 1500)
  }

  if (!analysis) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-gray-400 flex items-center gap-3"><RefreshCw className="animate-spin" size={20} />Loading…</div>
    </div>
  )

  const rc = RISK_CFG[analysis.risk_level] || RISK_CFG.Safe

  // ─── RENDER ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">

      {/* Sub-header */}
      <div className="bg-white border-b border-gray-200 px-4 py-2.5 shadow-sm">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/analyze')} className="text-gray-400 hover:text-gray-700 transition-colors"><ArrowLeft size={17} /></button>
            <Shield size={16} className="text-indigo-600" />
            <span className="font-semibold text-sm text-gray-800">Audit Report</span>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${rc.bg} ${rc.border} ${rc.text}`}>{analysis.risk_level}</span>
           
          </div>
          <div className="flex gap-2">
            <button onClick={() => setChatOpen(o => !o)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${chatOpen ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-indigo-600 border-indigo-200 hover:bg-indigo-50'}`}>
              <Bot size={13} />{chatOpen ? 'Close Chat' : 'AI Analysis Chat'}
            </button>
            <button onClick={exportJSON} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 text-xs transition-colors">
              <Download size={13} />Export
            </button>
            <button onClick={() => copyText(window.location.href, 'url')} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 text-xs transition-colors">
              {copied === 'url' ? <Check size={13} className="text-green-600" /> : <Share2 size={13} />}Share
            </button>
          </div>
        </div>
      </div>

      {/* MAIN LAYOUT */}
      <div className="max-w-screen-2xl mx-auto px-4 py-5 flex gap-5">
        <div className="flex-1 min-w-0 space-y-5">

          {/* OVERVIEW CARDS */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'Vulnerabilities', value: detectedVulns.length, icon: Bug, color: '#ef4444', bg: 'bg-red-50', sub: 'above threshold' },
              { label: 'Risk Level', value: analysis.risk_level, icon: AlertTriangle, color: rc.hex, bg: rc.bg, sub: 'overall severity' },
              { label: 'Functions at Risk', value: (analysis.function_metrics || []).filter(f => f.risk_score > 0.4).length, icon: Target, color: '#f97316', bg: 'bg-orange-50', sub: `of ${(analysis.function_metrics || []).length} functions` },
              { label: 'Attention Score', value: analysis.attention_weights?.length ? Math.round(Math.max(...analysis.attention_weights) * 100) + '%' : 'N/A', icon: Activity, color: '#8b5cf6', bg: 'bg-purple-50', sub: 'max GAT weight' },
            ].map((c, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}
                className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-gray-500 text-xs font-medium">{c.label}</span>
                  <div className={`w-7 h-7 rounded-lg ${c.bg} flex items-center justify-center`}>
                    <c.icon size={14} style={{ color: c.color }} />
                  </div>
                </div>
                <div className="text-2xl font-bold text-gray-900">{c.value}</div>
                <div className="text-gray-400 text-xs mt-1">{c.sub}</div>
              </motion.div>
            ))}
          </div>

          {/* RISK RING + VULN BARS */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm flex flex-col items-center gap-4">
              <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest">Overall Risk Score</h3>
              <RiskRing score={analysis.overall_risk_score} level={analysis.risk_level} />
              <p className="text-gray-500 text-xs text-center line-clamp-3">{analysis.summary}</p>
            </div>
            <div className="lg:col-span-2 bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
              <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4">Vulnerability Probabilities</h3>
              <ResponsiveContainer width="100%" height={195}>
                <BarChart layout="vertical" margin={{ left: 10 }}
                  data={(analysis.vulnerabilities || []).map(v => ({ name: v.type, prob: pct(v.probability) }))}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 10 }} unit="%" />
                  <YAxis dataKey="name" type="category" tick={{ fill: '#6b7280', fontSize: 11 }} width={110} />
                  <Tooltip formatter={v => [`${v}%`, 'Probability']} contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                    {(analysis.vulnerabilities || []).map((v, i) => <Cell key={i} fill={VULN_COLORS[v.type] || '#6b7280'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* TABS */}
          <div className="flex gap-1 bg-white rounded-xl p-1 border border-gray-200 shadow-sm">
            {[
              { id: 'overview', label: 'Charts',       icon: TrendingUp },
              { id: 'cfg',      label: 'CFG Graph',    icon: Network },
              { id: 'code',     label: 'Code Heatmap', icon: Code2 },
              { id: 'tables',   label: 'Tables',       icon: Table2 },
              { id: 'slither',  label: 'Slither',      icon: GitBranch, badge: slitherResult?.findings?.length },
              { id: 'mythril',  label: 'Mythril',      icon: Shield,    badge: mythrilResult?.findings?.length },
              { id: 'compare',  label: 'Compare',      icon: BarChart3 },
            ].map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-xs font-medium transition-all ${
                  activeTab === t.id ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}>
                <t.icon size={13} />
                <span className="hidden sm:inline">{t.label}</span>
                {t.badge > 0 && (
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${activeTab === t.id ? 'bg-white/30 text-white' : 'bg-red-100 text-red-600'}`}>{t.badge}</span>
                )}
                {staticLoading && (t.id === 'slither' || t.id === 'mythril') && !t.badge && (
                  <span className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                )}
              </button>
            ))}
          </div>

          {/* ── CHARTS ─────────────────────────────────────────────────── */}
          {activeTab === 'overview' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-1 lg:grid-cols-2 gap-4">

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4 flex items-center gap-2"><Activity size={13} />Vulnerability Radar</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <RadarChart data={analysis.radar_data || []}>
                    <PolarGrid stroke="#e5e7eb" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#6b7280', fontSize: 11 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 9 }} />
                    <Radar name="Risk %" dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} formatter={v => [`${v}%`]} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4 flex items-center gap-2"><Layers size={13} />Risk Distribution</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie data={pieData.length ? pieData : [{ name: 'Safe', value: 100, fill: '#16a34a' }]}
                      cx="50%" cy="50%" outerRadius={90} dataKey="value"
                      label={({ name, value }) => `${name.split(' ')[0]} ${value}%`} labelLine={{ stroke: '#d1d5db' }}>
                      {pieData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4 flex items-center gap-2"><TrendingUp size={13} />Line-by-Line Risk</h3>
                <div className="overflow-x-auto">
                  <div style={{ minWidth: Math.max(400, lineRiskData.length * 18) }}>
                    <ResponsiveContainer width="100%" height={190}>
                      <BarChart data={lineRiskData}>
                        <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 9 }} interval={Math.floor(lineRiskData.length / 8)} />
                        <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 9 }} unit="%" />
                        <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} formatter={v => [`${v}%`, 'Risk']} />
                        <Bar dataKey="risk" radius={[2, 2, 0, 0]}>
                          {lineRiskData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4 flex items-center gap-2"><Eye size={13} />Attention Timeline</h3>
                <ResponsiveContainer width="100%" height={190}>
                  <LineChart data={attentionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="window" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 10 }} unit="%" />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} formatter={v => [`${v}%`, 'Attention']} />
                    <Line type="monotone" dataKey="attention" stroke="#6366f1" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4 flex items-center gap-2"><Target size={13} />Risk Gauge per Vulnerability</h3>
                <ResponsiveContainer width="100%" height={210}>
                  <RadialBarChart cx="50%" cy="50%" innerRadius="15%" outerRadius="85%" data={radialData} startAngle={180} endAngle={0}>
                    <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                    <RadialBar label={{ position: 'insideStart', fill: '#fff', fontSize: 10 }} dataKey="value" background={{ fill: '#f9fafb' }} />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} formatter={v => [`${v}%`]} />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="text-gray-600 font-semibold text-xs uppercase tracking-widest mb-4 flex items-center gap-2"><Layers size={13} />Function Severity Treemap</h3>
                <ResponsiveContainer width="100%" height={210}>
                  <Treemap
                    data={(analysis.function_metrics || [{ name: 'contract', risk_score: analysis.overall_risk_score }]).map(f => ({
                      name: f.name, size: Math.max(10, Math.round((f.risk_score || 0.05) * 100)), fill: riskColor(f.risk_score || 0.05),
                    }))}
                    dataKey="size" stroke="#fff"
                    content={({ x, y, width, height, name, value, fill }) => (
                      width > 20 && height > 15 ? (
                        <g>
                          <rect x={x} y={y} width={width} height={height} fill={fill} stroke="#fff" strokeWidth={2} rx={4} />
                          <text x={x + width / 2} y={y + height / 2} textAnchor="middle" fill="#fff" fontSize={Math.min(12, width / 5)} fontWeight={700}>{name}</text>
                          {height > 28 && <text x={x + width / 2} y={y + height / 2 + 13} textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize={10}>{value}%</text>}
                        </g>
                      ) : null
                    )}
                  />
                </ResponsiveContainer>
              </div>

            </motion.div>
          )}

          {/* ── CFG GRAPH ─────────────────────────────────────────────── */}
          {activeTab === 'cfg' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="p-4 border-b border-gray-100 flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2"><Network size={15} className="text-indigo-600" />Control Flow Graph</h3>
                    <p className="text-gray-400 text-xs mt-0.5">Actual program execution flow · Click any node → jump to code · Drag to pan · Scroll to zoom</p>
                  </div>
                  {/* Legend */}
                  <div className="flex flex-wrap gap-2">
                    {[
                      ['▶','entry','#16a34a','ENTRY / EXIT'],
                      ['ƒ','function','#3b82f6','Function'],
                      ['⟐','branch','#eab308','if/else branch'],
                      ['↺','loop','#0ea5e9','for/while loop'],
                      ['⚠','external_call','#ef4444','External call'],
                      ['✎','state_write','#8b5cf6','State write'],
                      ['✓','guard','#22c55e','require/check'],
                    ].map(([icon, type, color, label]) => (
                      <span key={type} className="flex items-center gap-1 text-xs text-gray-500 bg-gray-50 border border-gray-200 rounded px-2 py-0.5">
                        <span style={{ color }}>{icon}</span>{label}
                      </span>
                    ))}
                  </div>
                  {/* Edge legend */}
                  <div className="w-full flex flex-wrap gap-3 text-xs text-gray-500 pt-1">
                    {[['#94a3b8','solid','Control flow'],['#22c55e','solid','True branch (T)'],['#ef4444','dashed','False branch (F)'],['#0ea5e9','animated','Loop back-edge ↺'],['#dc2626','animated thick','Risky path ⚠']].map(([c,s,l]) => (
                      <span key={l} className="flex items-center gap-1.5">
                        <span style={{ display: 'inline-block', width: 20, height: 2, background: c, borderRadius: 1, opacity: s === 'dashed' ? 0.7 : 1 }} />
                        {l}
                      </span>
                    ))}
                  </div>
                </div>

                <div style={{ height: 500 }}>
                  {rfNodes.length > 0 ? (
                    <ReactFlow nodes={rfNodes} edges={rfEdges}
                      onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
                      onNodeClick={handleNodeClick} nodeTypes={NODE_TYPES}
                      fitView fitViewOptions={{ padding: 0.2 }}
                      style={{ background: '#f8fafc' }}>
                      <Background color="#e2e8f0" gap={20} />
                      <Controls style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} />
                      <MiniMap
                        nodeColor={n => NODE_STYLES[n.data?.node_type]?.border || '#94a3b8'}
                        style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }} />
                    </ReactFlow>
                  ) : (
                    <div className="h-full flex items-center justify-center text-gray-400 flex-col gap-3">
                      <Network size={40} className="opacity-30" />
                      <p className="text-sm">No CFG data — re-run the analysis</p>
                    </div>
                  )}
                </div>

                {selectedNode && (
                  <div className="p-4 border-t border-gray-100 bg-gray-50">
                    <div className="flex items-center gap-3 flex-wrap">
                      <span className="font-bold text-gray-800">{selectedNode.label}</span>
                      <span className="text-gray-500 text-sm">· {selectedNode.node_type}</span>
                      {selectedNode.risk_score > 0.05 && (
                        <span className="font-bold text-sm" style={{ color: riskColor(selectedNode.risk_score) }}>
                          {Math.round(selectedNode.risk_score * 100)}% risk
                        </span>
                      )}
                      {selectedNode.lines?.length > 0 && (
                        <span className="ml-auto text-gray-400 text-xs font-mono">Lines: {selectedNode.lines.join(', ')}</span>
                      )}
                    </div>
                    {selectedNode.vuln_types?.length > 0 && (
                      <div className="mt-2 flex gap-2 flex-wrap">
                        {selectedNode.vuln_types.map(vt => (
                          <span key={vt} className="px-2 py-0.5 rounded-full text-xs font-medium"
                            style={{ background: (VULN_COLORS[vt] || '#6b7280') + '20', color: VULN_COLORS[vt] || '#6b7280', border: `1px solid ${(VULN_COLORS[vt] || '#6b7280')}40` }}>
                            {vt}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* ── CODE HEATMAP ──────────────────────────────────────────── */}
          {activeTab === 'code' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="p-4 border-b border-gray-100 flex flex-wrap items-center justify-between gap-3">
                  <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2"><Code2 size={15} className="text-indigo-600" />Line-by-Line Attention Heatmap</h3>
                  <div className="flex gap-3 text-xs text-gray-500">
                    {[['#16a34a','Safe'],['#65a30d','Low'],['#ca8a04','Medium'],['#ea580c','High'],['#dc2626','Critical']].map(([c,l]) => (
                      <span key={l} className="flex items-center gap-1"><span style={{ background: c }} className="w-3 h-1.5 rounded inline-block" />{l}</span>
                    ))}
                  </div>
                </div>
                <div className="overflow-auto max-h-[560px]">
                  <table className="w-full text-xs font-mono">
                    <tbody>
                      {(analysis.line_risks || []).map((lr) => {
                        const highlighted = selectedNode?.lines?.includes(lr.line_number)
                        return (
                          <tr key={lr.line_number}
                            className={`border-b border-gray-50 hover:bg-gray-50/80 ${highlighted ? 'bg-indigo-50 ring-2 ring-inset ring-indigo-300' : ''}`}
                            style={{ borderLeft: `3px solid ${lr.risk_score > 0.08 ? riskColor(lr.risk_score) : 'transparent'}` }}>
                            <td className="text-gray-300 px-3 py-1 select-none w-12 text-right">{lr.line_number}</td>
                            <td className="px-1 py-1 w-14">
                              {lr.risk_score > 0.05 && (
                                <div className="h-1.5 rounded-full" style={{ width: `${clamp(lr.risk_score * 100, 5, 100)}%`, background: riskColor(lr.risk_score) }} />
                              )}
                            </td>
                            <td className="px-3 py-1 whitespace-pre text-gray-700">{lr.content || ' '}</td>
                            <td className="px-3 py-1 w-16">
                              {lr.is_vulnerable && <span style={{ color: riskColor(lr.risk_score) }} className="font-bold">⚠ {Math.round(lr.risk_score * 100)}%</span>}
                            </td>
                            <td className="px-2 py-1 w-8">
                              <button onClick={() => copyText(lr.content, lr.line_number)} className="text-gray-300 hover:text-gray-600">
                                {copied === lr.line_number ? <Check size={11} className="text-green-600" /> : <Copy size={11} />}
                              </button>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2 mb-4">
                  <Sparkles size={14} className="text-yellow-500" />Security Recommendations
                </h3>
                <ul className="space-y-2">
                  {(analysis.recommendations || []).map((r, i) => (
                    <li key={i} className="flex items-start gap-3 text-gray-600 text-sm">
                      <span className="text-indigo-500 font-bold shrink-0">{i + 1}.</span>{r}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}

          {/* ── TABLES ─────────────────────────────────────────────────── */}
          {activeTab === 'tables' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">

              {/* Function Summary */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="p-4 border-b border-gray-100">
                  <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2"><FileText size={14} className="text-indigo-600" />Function Summary Table</h3>
                  <p className="text-gray-400 text-xs mt-0.5">Click column headers to sort</p>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-100">
                      {[['name','Function'],['risk_score','Risk %'],['top_vuln','Top Vuln'],['gat_score','GAT Score'],['loc','LoC'],['vulnerability_count','Vulns']].map(([col, label]) => (
                        <th key={col} onClick={() => sortBy(col)}
                          className="text-left px-4 py-3 text-gray-500 font-medium text-xs cursor-pointer hover:text-gray-800 select-none whitespace-nowrap">
                          {label}{sortCol === col && <span className="ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedFunctions.length === 0 ? (
                      <tr><td colSpan={6} className="text-center py-10 text-gray-400">Re-run analysis to generate function data</td></tr>
                    ) : sortedFunctions.map((fn, i) => (
                      <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/80">
                        <td className="px-4 py-3 font-mono text-gray-800 text-xs">{fn.name}</td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-gray-100 rounded-full h-1.5">
                              <div className="h-1.5 rounded-full" style={{ width: `${Math.min(100, Math.round(fn.risk_score * 100))}%`, background: riskColor(fn.risk_score) }} />
                            </div>
                            <span style={{ color: riskColor(fn.risk_score) }} className="text-xs font-bold">{Math.round(fn.risk_score * 100)}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-0.5 rounded-full text-xs font-medium" style={{ background: (VULN_COLORS[fn.top_vuln] || '#6b7280') + '15', color: VULN_COLORS[fn.top_vuln] || '#6b7280' }}>{fn.top_vuln}</span>
                        </td>
                        <td className="px-4 py-3 text-gray-500 font-mono text-xs">{fn.gat_score?.toFixed(3)}</td>
                        <td className="px-4 py-3 text-gray-500 text-xs">{fn.loc}</td>
                        <td className="px-4 py-3"><span className={`font-bold ${fn.vulnerability_count > 0 ? 'text-red-500' : 'text-green-600'}`}>{fn.vulnerability_count}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Detailed Findings */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="p-4 border-b border-gray-100">
                  <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2"><Bug size={14} className="text-indigo-600" />Detailed Findings</h3>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-100">
                      {['Vulnerability','SWC','Confidence','Probability','Affected Lines','Severity'].map(h => (
                        <th key={h} className="text-left px-4 py-3 text-gray-500 font-medium text-xs whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(analysis.vulnerabilities || []).map((v, i) => (
                      <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/80">
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full" style={{ background: VULN_COLORS[v.type] || '#6b7280' }} />
                            <span className="text-gray-800 font-medium">{v.type}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 font-mono text-gray-400 text-xs">{SWC_MAP[v.type] || '–'}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            v.confidence === 'High' ? 'bg-red-50 text-red-600' :
                            v.confidence === 'Medium' ? 'bg-yellow-50 text-yellow-700' : 'bg-gray-100 text-gray-500'}`}>
                            {v.confidence}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-mono font-bold text-sm" style={{ color: riskColor(v.probability) }}>{(v.probability * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-gray-400 text-xs font-mono">{v.affected_lines?.slice(0, 6).join(', ') || '–'}</td>
                        <td className="px-4 py-3">
                          <span className={`text-xs font-bold ${
                            v.probability >= 0.8 ? 'text-red-600' : v.probability >= 0.6 ? 'text-orange-600' :
                            v.probability >= 0.4 ? 'text-yellow-600' : 'text-gray-400'}`}>
                            {v.probability >= 0.8 ? 'Critical' : v.probability >= 0.6 ? 'High' : v.probability >= 0.4 ? 'Medium' : 'Low'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}

          {/* ── SLITHER ────────────────────────────────────────────────── */}
          {activeTab === 'slither' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
              {staticLoading ? (
                <div className="bg-white border border-gray-200 rounded-xl p-14 text-center shadow-sm">
                  <RefreshCw size={28} className="mx-auto text-indigo-400 animate-spin mb-3" />
                  <p className="text-gray-500 text-sm">Running Slither analysis…</p>
                </div>
              ) : slitherResult ? (
                <>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: 'Total Findings', value: slitherResult.findings?.length || 0, color: '#f97316', bg: 'bg-orange-50', sub: slitherResult.available ? 'Slither Python API' : 'Not available' },
                      { label: 'High / Critical', value: slitherResult.findings?.filter(f => f.impact === 'High').length || 0, color: '#dc2626', bg: 'bg-red-50', sub: 'Needs immediate fix' },
                      { label: 'Medium / Low', value: slitherResult.findings?.filter(f => f.impact !== 'High').length || 0, color: '#ca8a04', bg: 'bg-yellow-50', sub: 'Review recommended' },
                    ].map((c, i) => (
                      <div key={i} className={`${c.bg} border border-gray-200 rounded-xl p-4 text-center shadow-sm`}>
                        <div className="text-3xl font-bold" style={{ color: c.color }}>{c.value}</div>
                        <div className="text-gray-700 text-sm font-semibold mt-1">{c.label}</div>
                        <div className="text-gray-400 text-xs">{c.sub}</div>
                      </div>
                    ))}
                  </div>
                  {slitherResult.error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">{slitherResult.error}</div>
                  )}
                  {slitherResult.findings?.length > 0 ? (
                    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                      <div className="p-4 border-b border-gray-100 flex items-center gap-2">
                        <GitBranch size={14} className="text-orange-500" />
                        <h3 className="font-semibold text-gray-800 text-sm">Slither Findings</h3>
                        <span className="text-xs text-gray-400 ml-auto">Real static analysis — no simulation</span>
                      </div>
                      <div className="divide-y divide-gray-50">
                        {slitherResult.findings.map((f, i) => (
                          <div key={i} className="p-4 hover:bg-gray-50/60">
                            <div className="flex flex-wrap items-center gap-2 mb-1">
                              <span className="font-mono text-xs bg-gray-100 border border-gray-200 px-2 py-0.5 rounded text-gray-700">{f.check}</span>
                              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                                f.impact === 'High' ? 'bg-red-50 text-red-700' :
                                f.impact === 'Medium' ? 'bg-orange-50 text-orange-700' :
                                f.impact === 'Low' ? 'bg-yellow-50 text-yellow-700' : 'bg-gray-100 text-gray-500'}`}>{f.impact}</span>
                              <span className="text-gray-400 text-xs">{f.confidence} confidence</span>
                              {f.lines?.length > 0 && <span className="ml-auto text-gray-400 text-xs font-mono">L{f.lines.join(', ')}</span>}
                            </div>
                            <p className="text-gray-600 text-xs leading-relaxed">{f.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-white border border-gray-200 rounded-xl p-10 text-center shadow-sm">
                      <CheckCircle size={32} className="mx-auto text-green-400 mb-3" />
                      <p className="text-gray-600 font-medium">No Slither findings</p>
                      <p className="text-gray-400 text-xs mt-1">Contract passed all Slither checks</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-white border border-gray-200 rounded-xl p-14 text-center shadow-sm">
                  <GitBranch size={36} className="mx-auto text-gray-300 mb-4" />
                  <p className="text-gray-500">Load a contract to run Slither analysis.</p>
                </div>
              )}
            </motion.div>
          )}

          {/* ── MYTHRIL ────────────────────────────────────────────────── */}
          {activeTab === 'mythril' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
              {staticLoading ? (
                <div className="bg-white border border-gray-200 rounded-xl p-14 text-center shadow-sm">
                  <RefreshCw size={28} className="mx-auto text-purple-400 animate-spin mb-3" />
                  <p className="text-gray-500 text-sm">Running Mythril SWC checks…</p>
                </div>
              ) : mythrilResult ? (
                <>
                  <div className="grid grid-cols-4 gap-4">
                    {['Critical','High','Medium','Low'].map(sev => {
                      const count = mythrilResult.findings?.filter(f => f.severity === sev).length || 0
                      const colors = { Critical: ['#dc2626','bg-red-50'], High: ['#ea580c','bg-orange-50'], Medium: ['#ca8a04','bg-yellow-50'], Low: ['#65a30d','bg-lime-50'] }
                      const [color, bg] = colors[sev]
                      return (
                        <div key={sev} className={`${bg} border border-gray-200 rounded-xl p-4 text-center shadow-sm`}>
                          <div className="text-3xl font-bold" style={{ color }}>{count}</div>
                          <div className="text-gray-700 text-sm font-semibold mt-1">{sev}</div>
                        </div>
                      )
                    })}
                  </div>
                  {mythrilResult.findings?.length > 0 ? (
                    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                      <div className="p-4 border-b border-gray-100 flex items-center gap-2">
                        <Shield size={14} className="text-purple-600" />
                        <h3 className="font-semibold text-gray-800 text-sm">Mythril SWC Findings</h3>
                        <span className="text-xs text-gray-400 ml-auto">Symbolic execution checks — real analysis</span>
                      </div>
                      <div className="divide-y divide-gray-50">
                        {mythrilResult.findings.map((f, i) => (
                          <div key={i} className="p-4 hover:bg-gray-50/60">
                            <div className="flex flex-wrap items-center gap-2 mb-1.5">
                              <span className="font-mono text-xs bg-purple-50 border border-purple-100 px-2 py-0.5 rounded text-purple-700">{f.swc}</span>
                              <span className="font-semibold text-gray-800 text-sm">{f.title}</span>
                              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                                f.severity === 'Critical' ? 'bg-red-50 text-red-700' :
                                f.severity === 'High' ? 'bg-orange-50 text-orange-700' :
                                f.severity === 'Medium' ? 'bg-yellow-50 text-yellow-700' : 'bg-gray-100 text-gray-500'}`}>{f.severity}</span>
                              <span className="text-gray-400 text-xs">{f.confidence}</span>
                              {f.lines?.length > 0 && <span className="ml-auto text-gray-400 text-xs font-mono">L{f.lines.slice(0,6).join(', ')}</span>}
                            </div>
                            <p className="text-gray-600 text-xs leading-relaxed">{f.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-white border border-gray-200 rounded-xl p-10 text-center shadow-sm">
                      <CheckCircle size={32} className="mx-auto text-green-400 mb-3" />
                      <p className="text-gray-600 font-medium">No Mythril findings</p>
                      <p className="text-gray-400 text-xs mt-1">Contract passed all SWC checks</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-white border border-gray-200 rounded-xl p-14 text-center shadow-sm">
                  <Shield size={36} className="mx-auto text-gray-300 mb-4" />
                  <p className="text-gray-500">Load a contract to run Mythril SWC analysis.</p>
                </div>
              )}
            </motion.div>
          )}

          {/* ── TRI-TOOL COMPARISON ───────────────────────────────────── */}
          {activeTab === 'compare' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">

              {/* Summary badges */}
              <div className="grid grid-cols-4 gap-4">
                {[
                  { label: 'All 3 Agree',   value: triComparison.filter(r => r.consensus === 'all').length,  color: '#16a34a', bg: 'bg-green-50',  sub: 'High confidence' },
                  { label: 'Two Agree',      value: triComparison.filter(r => r.consensus === 'two').length,  color: '#ca8a04', bg: 'bg-yellow-50', sub: 'Moderate confidence' },
                  { label: 'Model Only',     value: triComparison.filter(r => r.modelHit && !r.slitherHit && !r.mythrilHit).length, color: '#6366f1', bg: 'bg-indigo-50', sub: 'GNN exclusive' },
                  { label: 'Not Detected',   value: triComparison.filter(r => r.consensus === 'none').length, color: '#94a3b8', bg: 'bg-gray-50',   sub: 'All tools clean' },
                ].map((c, i) => (
                  <div key={i} className={`${c.bg} border border-gray-200 rounded-xl p-4 text-center shadow-sm`}>
                    <div className="text-3xl font-bold" style={{ color: c.color }}>{c.value}</div>
                    <div className="text-gray-700 text-sm font-semibold mt-1">{c.label}</div>
                    <div className="text-gray-400 text-xs">{c.sub}</div>
                  </div>
                ))}
              </div>

              {/* Main comparison table */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="p-4 border-b border-gray-100 flex items-center gap-3">
                  <BarChart3 size={15} className="text-indigo-600" />
                  <h3 className="font-semibold text-gray-800 text-sm">Model vs Slither vs Mythril</h3>
                  <span className="text-xs text-gray-400 ml-auto">
                    {staticLoading ? <span className="flex items-center gap-1"><RefreshCw size={11} className="animate-spin" />Running tools…</span> : 'Real findings — no simulation'}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200">
                        <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs w-36">Vulnerability</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs w-16">SWC</th>
                        <th className="text-center px-4 py-3 text-indigo-600 font-semibold text-xs">
                          <div className="flex flex-col items-center gap-0.5">
                            <span>Our Model</span>
                            <span className="text-indigo-300 font-normal text-[10px]">Hybrid GNN</span>
                          </div>
                        </th>
                        <th className="text-center px-4 py-3 text-orange-600 font-semibold text-xs">
                          <div className="flex flex-col items-center gap-0.5">
                            <span>Slither</span>
                            <span className="text-orange-300 font-normal text-[10px]">{slitherResult?.available ? 'Python API' : 'N/A'}</span>
                          </div>
                        </th>
                        <th className="text-center px-4 py-3 text-purple-600 font-semibold text-xs">
                          <div className="flex flex-col items-center gap-0.5">
                            <span>Mythril</span>
                            <span className="text-purple-300 font-normal text-[10px]">SWC checks</span>
                          </div>
                        </th>
                        <th className="text-center px-4 py-3 text-gray-500 font-medium text-xs">Consensus</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium text-xs">Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {triComparison.map((row, i) => (
                        <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/60">
                          {/* Vulnerability name */}
                          <td className="px-4 py-4">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: VULN_COLORS[row.name] || '#6b7280' }} />
                              <span className="font-semibold text-gray-800 text-xs">{row.name}</span>
                            </div>
                          </td>

                          {/* SWC */}
                          <td className="px-4 py-4">
                            <span className="font-mono text-xs text-gray-400">{row.swc}</span>
                          </td>

                          {/* Model */}
                          <td className="px-4 py-4 text-center">
                            {row.modelHit ? (
                              <div className="flex flex-col items-center gap-1">
                                <span className="text-indigo-600 font-bold text-sm">✓</span>
                                <span className="text-xs font-bold" style={{ color: riskColor(row.modelProb) }}>
                                  {Math.round(row.modelProb * 100)}%
                                </span>
                              </div>
                            ) : (
                              <div className="flex flex-col items-center gap-1">
                                <span className="text-gray-200 text-sm">–</span>
                                <span className="text-xs text-gray-300">{Math.round(row.modelProb * 100)}%</span>
                              </div>
                            )}
                          </td>

                          {/* Slither */}
                          <td className="px-4 py-4 text-center">
                            {staticLoading ? (
                              <span className="w-3 h-3 border border-orange-300 border-t-transparent rounded-full animate-spin inline-block" />
                            ) : row.slitherHit ? (
                              <div className="flex flex-col items-center gap-1">
                                <span className="text-orange-500 font-bold text-sm">✓</span>
                                <span className="text-xs text-orange-400">{row.slitherFindings[0]?.impact || 'Found'}</span>
                              </div>
                            ) : (
                              <span className="text-gray-200 text-sm">–</span>
                            )}
                          </td>

                          {/* Mythril */}
                          <td className="px-4 py-4 text-center">
                            {staticLoading ? (
                              <span className="w-3 h-3 border border-purple-300 border-t-transparent rounded-full animate-spin inline-block" />
                            ) : row.mythrilHit ? (
                              <div className="flex flex-col items-center gap-1">
                                <span className="text-purple-600 font-bold text-sm">✓</span>
                                <span className="text-xs text-purple-400">{row.mythrilFindings[0]?.severity || 'Found'}</span>
                              </div>
                            ) : (
                              <span className="text-gray-200 text-sm">–</span>
                            )}
                          </td>

                          {/* Consensus badge */}
                          <td className="px-4 py-4 text-center">
                            {row.consensus === 'all' && (
                              <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-green-50 text-green-700 border border-green-200 whitespace-nowrap">✓ All Agree</span>
                            )}
                            {row.consensus === 'two' && (
                              <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-yellow-50 text-yellow-700 border border-yellow-200 whitespace-nowrap">2/3 Agree</span>
                            )}
                            {row.consensus === 'one' && row.modelHit && (
                              <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200 whitespace-nowrap">Model Only</span>
                            )}
                            {row.consensus === 'one' && row.slitherHit && !row.modelHit && (
                              <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-orange-50 text-orange-700 border border-orange-200 whitespace-nowrap">Slither Only</span>
                            )}
                            {row.consensus === 'one' && row.mythrilHit && !row.modelHit && !row.slitherHit && (
                              <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-purple-50 text-purple-700 border border-purple-200 whitespace-nowrap">Mythril Only</span>
                            )}
                            {row.consensus === 'none' && (
                              <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-gray-50 text-gray-400 border border-gray-100 whitespace-nowrap">Not Detected</span>
                            )}
                          </td>

                          {/* Details snippet */}
                          <td className="px-4 py-4 max-w-xs">
                            {row.slitherFindings[0] && (
                              <p className="text-xs text-orange-600 truncate" title={row.slitherFindings[0].description}>
                                <span className="font-mono text-[10px] bg-orange-50 px-1 rounded mr-1">{row.slitherFindings[0].check}</span>
                                {row.slitherFindings[0].description?.slice(0, 60)}…
                              </p>
                            )}
                            {row.mythrilFindings[0] && (
                              <p className="text-xs text-purple-600 truncate mt-0.5" title={row.mythrilFindings[0].description}>
                                <span className="font-mono text-[10px] bg-purple-50 px-1 rounded mr-1">{row.mythrilFindings[0].swc}</span>
                                {row.mythrilFindings[0].description?.slice(0, 60)}…
                              </p>
                            )}
                            {!row.slitherFindings[0] && !row.mythrilFindings[0] && row.modelHit && (
                              <p className="text-xs text-indigo-400 italic">Model detected — static tools clean</p>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* All raw Slither + Mythril findings not mapped to the 5 categories */}
              {(() => {
                const mappedSlither = new Set(triComparison.flatMap(r => r.slitherFindings.map(f => f.check)))
                const mappedMythril = new Set(triComparison.flatMap(r => r.mythrilFindings.map(f => f.swc)))
                const extraSlither = (slitherResult?.findings || []).filter(f => !mappedSlither.has(f.check))
                const extraMythril = (mythrilResult?.findings  || []).filter(f => !mappedMythril.has(f.swc))
                if (!extraSlither.length && !extraMythril.length) return null
                return (
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                    <div className="p-4 border-b border-gray-100">
                      <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2">
                        <AlertTriangle size={14} className="text-amber-500" />
                        Additional Findings (outside 5 core categories)
                      </h3>
                    </div>
                    <div className="divide-y divide-gray-50">
                      {extraSlither.map((f, i) => (
                        <div key={`s${i}`} className="p-3 flex gap-3 items-start">
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-orange-50 text-orange-600 border border-orange-100 shrink-0">Slither</span>
                          <div className="min-w-0">
                            <span className="font-mono text-xs text-gray-600 mr-2">{f.check}</span>
                            <span className={`text-xs font-bold mr-2 ${f.impact==='High'?'text-red-600':f.impact==='Medium'?'text-orange-600':'text-gray-400'}`}>{f.impact}</span>
                            <p className="text-xs text-gray-500 truncate">{f.description}</p>
                          </div>
                        </div>
                      ))}
                      {extraMythril.map((f, i) => (
                        <div key={`m${i}`} className="p-3 flex gap-3 items-start">
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-50 text-purple-600 border border-purple-100 shrink-0">Mythril</span>
                          <div className="min-w-0">
                            <span className="font-mono text-xs text-gray-600 mr-2">{f.swc}</span>
                            <span className="font-semibold text-xs text-gray-700 mr-2">{f.title}</span>
                            <span className={`text-xs font-bold ${f.severity==='High'||f.severity==='Critical'?'text-red-600':f.severity==='Medium'?'text-orange-600':'text-gray-400'}`}>{f.severity}</span>
                            <p className="text-xs text-gray-500 truncate">{f.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })()}

            </motion.div>
          )}

        </div>{/* end main column */}

        {/* ── ANALYSIS-SPECIFIC CHAT SIDEBAR ─────────────────────────── */}
        <AnimatePresence>
          {chatOpen && (
            <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 360, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }} transition={{ duration: 0.22, ease: 'easeInOut' }}
              className="shrink-0 overflow-hidden">
              <div className="w-[360px] flex flex-col bg-white border border-gray-200 rounded-xl overflow-hidden shadow-lg"
                   style={{ maxHeight: 'calc(100vh - 130px)', position: 'sticky', top: 80 }}>

                <div className="px-4 py-3 bg-indigo-600 flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                    <Bot size={15} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="text-white font-semibold text-sm flex items-center gap-1.5">
                      Audit Assistant <Sparkles size={11} className="text-yellow-300" />
                    </div>
                    <div className="text-indigo-200 text-xs">Contract context auto-injected</div>
                  </div>
                  <button onClick={() => setChatOpen(false)} className="text-white/70 hover:text-white"><XCircle size={16} /></button>
                </div>

                <div className="px-3 py-2 border-b border-gray-100 flex gap-1.5 overflow-x-auto scrollbar-none bg-gray-50">
                  {['How to fix reentrancy?','Explain findings','Best practices','What is SWC-107?'].map(q => (
                    <button key={q} onClick={() => setChatInput(q)}
                      className="shrink-0 px-2.5 py-1 rounded-full bg-white border border-gray-200 text-gray-600 text-xs hover:border-indigo-300 hover:text-indigo-600 transition-colors whitespace-nowrap">
                      {q}
                    </button>
                  ))}
                </div>

                <div className="flex-1 overflow-y-auto p-3 space-y-3" style={{ minHeight: 0 }}>
                  {chatMsgs.map((m, i) => (
                    <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      {m.role === 'assistant' && (
                        <div className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center shrink-0 mt-0.5">
                          <Bot size={12} className="text-indigo-600" />
                        </div>
                      )}
                      <div className={`max-w-[88%] rounded-xl px-3 py-2 ${
                        m.role === 'user' ? 'bg-indigo-600 text-white text-xs rounded-br-none' : 'bg-gray-100 rounded-bl-none'
                      }`}>
                        {m.role === 'assistant' ? <MD text={m.content} /> : <span className="text-xs">{m.content}</span>}
                      </div>
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="flex gap-2">
                      <div className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center shrink-0">
                        <Bot size={12} className="text-indigo-600" />
                      </div>
                      <div className="bg-gray-100 rounded-xl px-3 py-2.5 flex gap-1 items-center">
                        {[0,1,2].map(i => <span key={i} className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: `${i*0.15}s` }} />)}
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div className="p-3 border-t border-gray-100">
                  <div className="flex gap-2">
                    <input value={chatInput} onChange={e => setChatInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendChatMsg()}
                      placeholder="Ask about this contract…"
                      className="flex-1 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-xs text-gray-800 placeholder-gray-400 focus:outline-none focus:border-indigo-400 focus:bg-white transition-colors" />
                    <button onClick={sendChatMsg} disabled={chatLoading || !chatInput.trim()}
                      className="w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 flex items-center justify-center shrink-0">
                      <Send size={13} className="text-white" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  )
}
