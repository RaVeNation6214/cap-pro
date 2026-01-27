import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Shield,
  Zap,
  Eye,
  Lock,
  Code,
  BarChart3,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Bug,
  GitBranch
} from 'lucide-react'
import { Button, Card, CardContent, Badge } from '../components/ui'

// Floating code snippets for background
const codeSnippets = [
  'function withdraw() {',
  'msg.sender.call{value: amount}("")',
  'require(tx.origin == owner)',
  'balances[msg.sender] -= amount;',
  'delegatecall(target, data)',
  'mapping(address => uint256)',
]

// Stats for counter animation
const stats = [
  { label: 'Vulnerabilities Detected', value: 15000, suffix: '+' },
  { label: 'Contracts Analyzed', value: 50000, suffix: '+' },
  { label: 'Accuracy Rate', value: 89, suffix: '%' },
  { label: 'Detection Classes', value: 4, suffix: '' },
]

// Features
const features = [
  {
    icon: Shield,
    title: 'Multi-Class Detection',
    description: 'Identifies 4 critical vulnerability types: Reentrancy, Access Control, Arithmetic, and Unchecked Calls.',
    color: 'primary',
  },
  {
    icon: Eye,
    title: 'Explainable AI',
    description: 'Attention-based visualization shows exactly which code regions pose security risks.',
    color: 'accent',
  },
  {
    icon: Zap,
    title: 'Lightning Fast',
    description: 'Analyze contracts in seconds with our optimized Hierarchical Transformer architecture.',
    color: 'primary',
  },
  {
    icon: Lock,
    title: 'Enterprise Ready',
    description: 'Production-grade security analysis for DeFi protocols and smart contract auditing.',
    color: 'accent',
  },
]

// Vulnerability types showcase
const vulnerabilityTypes = [
  {
    name: 'Reentrancy',
    description: 'External call before state update',
    severity: 'critical',
    icon: Bug,
  },
  {
    name: 'Access Control',
    description: 'tx.origin authentication bypass',
    severity: 'critical',
    icon: Lock,
  },
  {
    name: 'Unchecked Calls',
    description: 'Missing return value checks',
    severity: 'high',
    icon: AlertTriangle,
  },
  {
    name: 'Arithmetic',
    description: 'Integer overflow/underflow',
    severity: 'high',
    icon: BarChart3,
  },
]

// Animated counter hook
function useCounter(end, duration = 2000) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let startTime = null
    const step = (timestamp) => {
      if (!startTime) startTime = timestamp
      const progress = Math.min((timestamp - startTime) / duration, 1)
      setCount(Math.floor(progress * end))
      if (progress < 1) {
        requestAnimationFrame(step)
      }
    }
    requestAnimationFrame(step)
  }, [end, duration])

  return count
}

function StatCounter({ stat, index }) {
  const count = useCounter(stat.value, 2000 + index * 200)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      viewport={{ once: true }}
      className="text-center"
    >
      <div className="text-4xl md:text-5xl font-bold gradient-text mb-2">
        {count.toLocaleString()}{stat.suffix}
      </div>
      <div className="text-dark-400">{stat.label}</div>
    </motion.div>
  )
}

function FloatingCode({ snippet, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 100 }}
      animate={{
        opacity: [0, 0.6, 0.6, 0],
        y: [-20, -100 - Math.random() * 100],
        x: Math.sin(index) * 50,
      }}
      transition={{
        duration: 8 + Math.random() * 4,
        repeat: Infinity,
        delay: index * 1.5,
        ease: 'easeInOut',
      }}
      className="absolute font-mono text-xs text-primary-500/30 whitespace-nowrap pointer-events-none"
      style={{
        left: `${10 + (index * 15) % 80}%`,
        bottom: '10%',
      }}
    >
      {snippet}
    </motion.div>
  )
}

export default function Home() {
  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          {codeSnippets.map((snippet, index) => (
            <FloatingCode key={index} snippet={snippet} index={index} />
          ))}

          {/* Grid pattern */}
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(14, 165, 233, 0.3) 1px, transparent 0)',
              backgroundSize: '40px 40px',
            }}
          />

          {/* Gradient orbs */}
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3],
            }}
            transition={{ duration: 8, repeat: Infinity }}
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/20 rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1.2, 1, 1.2],
              opacity: [0.3, 0.5, 0.3],
            }}
            transition={{ duration: 8, repeat: Infinity, delay: 1 }}
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-500/20 rounded-full blur-3xl"
          />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Badge variant="primary" size="lg" className="mb-6">
                <Zap className="w-4 h-4 mr-2" />
                AI-Powered Security Analysis
              </Badge>
            </motion.div>

            {/* Main heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-5xl md:text-7xl font-bold mb-6"
            >
              <span className="text-dark-100">Secure Your</span>
              <br />
              <span className="gradient-text">Smart Contracts</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-xl md:text-2xl text-dark-400 max-w-3xl mx-auto mb-8"
            >
              Detect vulnerabilities with our Hierarchical Transformer architecture.
              Get line-by-line risk analysis with explainable AI attention heatmaps.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link to="/analyze">
                <Button size="lg" icon={Zap}>
                  Start Analysis
                </Button>
              </Link>
              <Link to="/about">
                <Button variant="secondary" size="lg">
                  Learn More
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
            </motion.div>

            {/* Floating shield animation */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
              className="mt-16 relative"
            >
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                className="inline-block"
              >
                <div className="relative">
                  <Shield className="w-32 h-32 text-primary-400" />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="absolute inset-0 bg-primary-400/20 blur-2xl rounded-full"
                  />
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 border-y border-dark-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <StatCounter key={stat.label} stat={stat} index={index} />
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <Badge variant="secondary" className="mb-4">Features</Badge>
            <h2 className="text-4xl font-bold text-dark-100 mb-4">
              Why Choose SmartAudit?
            </h2>
            <p className="text-dark-400 max-w-2xl mx-auto">
              State-of-the-art vulnerability detection powered by our custom
              Hierarchical Transformer architecture.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full" hover>
                  <CardContent>
                    <motion.div
                      whileHover={{ scale: 1.1, rotate: 5 }}
                      className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${
                        feature.color === 'primary'
                          ? 'bg-primary-500/20 text-primary-400'
                          : 'bg-accent-500/20 text-accent-400'
                      }`}
                    >
                      <feature.icon className="w-6 h-6" />
                    </motion.div>
                    <h3 className="text-lg font-semibold text-dark-100 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-dark-400 text-sm">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Vulnerability Types Section */}
      <section className="py-20 bg-dark-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <Badge variant="danger" className="mb-4">Detection</Badge>
            <h2 className="text-4xl font-bold text-dark-100 mb-4">
              4 Critical Vulnerability Classes
            </h2>
            <p className="text-dark-400 max-w-2xl mx-auto">
              Our model is trained to detect the most common and dangerous
              smart contract vulnerabilities.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {vulnerabilityTypes.map((vuln, index) => (
              <motion.div
                key={vuln.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full border-l-4 border-l-red-500" hover>
                  <CardContent>
                    <div className="flex items-center justify-between mb-4">
                      <vuln.icon className="w-8 h-8 text-red-400" />
                      <Badge variant={vuln.severity}>
                        {vuln.severity}
                      </Badge>
                    </div>
                    <h3 className="text-lg font-semibold text-dark-100 mb-2">
                      {vuln.name}
                    </h3>
                    <p className="text-dark-400 text-sm">
                      {vuln.description}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="glass-card text-center p-12 gradient-border"
          >
            <motion.div
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity }}
            >
              <Shield className="w-16 h-16 text-primary-400 mx-auto mb-6" />
            </motion.div>
            <h2 className="text-3xl font-bold text-dark-100 mb-4">
              Ready to Secure Your Contracts?
            </h2>
            <p className="text-dark-400 mb-8 max-w-xl mx-auto">
              Upload your Solidity code and get instant vulnerability analysis
              with detailed explanations and recommendations.
            </p>
            <Link to="/analyze">
              <Button size="lg" icon={Zap}>
                Analyze Now
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
