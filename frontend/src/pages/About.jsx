import { motion } from 'framer-motion'
import {
  Shield,
  Brain,
  Layers,
  Eye,
  Target,
  Zap,
  Github,
  BookOpen,
  Users,
  Award,
  ChevronRight
} from 'lucide-react'
import { Button, Card, CardContent, Badge } from '../components/ui'
import { Link } from 'react-router-dom'

// Architecture layers
const architectureLayers = [
  {
    name: 'Input Layer',
    description: 'Solidity contract tokenization and windowing',
    icon: '📝',
    details: 'Splits contracts into 3-line overlapping windows for fine-grained analysis'
  },
  {
    name: 'Window Encoder',
    description: 'Transformer encoder for local patterns',
    icon: '🔍',
    details: '3 layers, 8 attention heads, d_model=256'
  },
  {
    name: 'Static Features',
    description: 'Structural cue extraction',
    icon: '📊',
    details: '12-dimensional feature vector per window'
  },
  {
    name: 'Attention Pooling',
    description: 'Contract-level aggregation',
    icon: '🎯',
    details: 'Learnable CLS token attends over all windows'
  },
  {
    name: 'Classifier',
    description: 'Multi-label prediction head',
    icon: '✅',
    details: '4-class sigmoid output with attention weights'
  }
]

// Team members (placeholder)
const team = [
  { name: 'Student 1', role: 'ML Architecture', avatar: '👨‍💻' },
  { name: 'Student 2', role: 'Frontend Development', avatar: '👩‍💻' },
  { name: 'Student 3', role: 'Backend & API', avatar: '👨‍🔬' },
]

// Research references
const references = [
  {
    title: 'SCVDIE: Ensemble Deep Learning',
    year: '2022',
    venue: 'Sensors',
    contribution: 'Inspired ensemble architecture approach'
  },
  {
    title: 'Lightning Cat: CodeBERT Functions',
    year: '2023',
    venue: 'ArXiv',
    contribution: 'Function-level analysis insights'
  },
  {
    title: 'GNNSE: GNN + Symbolic Execution',
    year: '2025',
    venue: 'Conference',
    contribution: 'State-of-the-art comparison baseline'
  }
]

export default function About() {
  return (
    <div className="pt-24 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <Badge variant="primary" className="mb-4">
            <BookOpen className="w-4 h-4 mr-2" />
            About the Project
          </Badge>
          <h1 className="text-4xl md:text-5xl font-bold text-dark-100 mb-6">
            Hierarchical Transformer for
            <br />
            <span className="gradient-text">Smart Contract Security</span>
          </h1>
          <p className="text-xl text-dark-400 max-w-3xl mx-auto">
            A B.Tech Capstone project exploring AI-driven vulnerability detection
            using a novel window-to-contract attention architecture.
          </p>
        </motion.div>

        {/* Problem Statement */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <Card hover={false} className="p-8">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div>
                <Badge variant="danger" className="mb-4">The Problem</Badge>
                <h2 className="text-2xl font-bold text-dark-100 mb-4">
                  Smart Contract Vulnerabilities Cost Billions
                </h2>
                <ul className="space-y-3 text-dark-400">
                  <li className="flex items-start gap-2">
                    <ChevronRight className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <span><strong className="text-dark-200">DAO Attack (2016):</strong> $50M stolen via reentrancy</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ChevronRight className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <span><strong className="text-dark-200">Parity (2017):</strong> $30M lost via delegatecall vulnerability</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ChevronRight className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <span><strong className="text-dark-200">2024-2025:</strong> Hundreds of ongoing DeFi exploits annually</span>
                  </li>
                </ul>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: 'Static Tools', issue: 'Miss complex patterns' },
                  { label: 'Early DL', issue: 'Black-box, no localization' },
                  { label: 'LLMs', issue: 'Expensive, require huge data' },
                  { label: 'Our Solution', issue: 'Practical & explainable', highlight: true },
                ].map((item, index) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, scale: 0.9 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className={`p-4 rounded-xl text-center ${
                      item.highlight
                        ? 'bg-primary-500/20 border border-primary-500/30'
                        : 'bg-dark-800/50 border border-dark-700'
                    }`}
                  >
                    <div className={`font-semibold mb-1 ${item.highlight ? 'text-primary-400' : 'text-dark-200'}`}>
                      {item.label}
                    </div>
                    <div className="text-sm text-dark-400">{item.issue}</div>
                  </motion.div>
                ))}
              </div>
            </div>
          </Card>
        </motion.section>

        {/* Architecture */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="text-center mb-8">
            <Badge variant="secondary" className="mb-4">
              <Layers className="w-4 h-4 mr-2" />
              Architecture
            </Badge>
            <h2 className="text-3xl font-bold text-dark-100">
              Hierarchical Transformer Design
            </h2>
          </div>

          <div className="relative">
            {/* Connection lines */}
            <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary-500 via-accent-500 to-primary-500 opacity-30 hidden md:block" />

            <div className="space-y-6">
              {architectureLayers.map((layer, index) => (
                <motion.div
                  key={layer.name}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex items-center gap-6 ${
                    index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'
                  }`}
                >
                  <div className={`flex-1 ${index % 2 === 0 ? 'md:text-right' : ''}`}>
                    <Card hover className="inline-block max-w-md">
                      <CardContent>
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-2xl">{layer.icon}</span>
                          <h3 className="text-lg font-semibold text-dark-100">
                            {layer.name}
                          </h3>
                        </div>
                        <p className="text-dark-400 text-sm mb-2">
                          {layer.description}
                        </p>
                        <code className="text-xs text-primary-400 bg-primary-500/10 px-2 py-1 rounded">
                          {layer.details}
                        </code>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="hidden md:flex items-center justify-center w-12 h-12 rounded-full bg-dark-800 border-2 border-primary-500/50 z-10">
                    <span className="text-primary-400 font-bold">{index + 1}</span>
                  </div>
                  <div className="flex-1 hidden md:block" />
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Key Features */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="text-center mb-8">
            <Badge variant="primary" className="mb-4">Features</Badge>
            <h2 className="text-3xl font-bold text-dark-100">
              What Makes Us Different
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: Eye,
                title: 'Explainable AI',
                description: 'Attention weights show exactly which code regions are risky, enabling developers to understand and fix issues.',
                color: 'primary'
              },
              {
                icon: Target,
                title: 'Fine-grained Detection',
                description: 'Window-level analysis provides line-by-line vulnerability localization, not just contract-level predictions.',
                color: 'accent'
              },
              {
                icon: Zap,
                title: 'Lightweight & Fast',
                description: '1.5-2M parameters only. Trains in hours, not days. No expensive LLMs or symbolic execution needed.',
                color: 'primary'
              }
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full" hover>
                  <CardContent>
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${
                      feature.color === 'primary'
                        ? 'bg-primary-500/20 text-primary-400'
                        : 'bg-accent-500/20 text-accent-400'
                    }`}>
                      <feature.icon className="w-6 h-6" />
                    </div>
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
        </motion.section>

        {/* Research Contributions */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="text-center mb-8">
            <Badge variant="secondary" className="mb-4">
              <Award className="w-4 h-4 mr-2" />
              Research
            </Badge>
            <h2 className="text-3xl font-bold text-dark-100">
              Novel Contributions
            </h2>
          </div>

          <Card hover={false}>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold text-dark-200 mb-4">
                    Our Innovations
                  </h3>
                  <ul className="space-y-3">
                    {[
                      'First systematic use of fault-line aligned windows for vulnerability detection',
                      'Window-to-contract hierarchical attention for interpretability',
                      'Lightweight syntax+semantic fusion without heavy GNN/CFG extraction',
                      'Comprehensive baseline comparison with transparent evaluation'
                    ].map((item, index) => (
                      <motion.li
                        key={index}
                        initial={{ opacity: 0, x: -10 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-start gap-2 text-dark-400"
                      >
                        <ChevronRight className="w-5 h-5 text-primary-400 flex-shrink-0 mt-0.5" />
                        <span>{item}</span>
                      </motion.li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-dark-200 mb-4">
                    Related Work
                  </h3>
                  <div className="space-y-3">
                    {references.map((ref, index) => (
                      <motion.div
                        key={ref.title}
                        initial={{ opacity: 0, x: 10 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: index * 0.1 }}
                        className="p-3 rounded-lg bg-dark-800/50"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-dark-200 text-sm">
                            {ref.title}
                          </span>
                          <Badge size="sm">{ref.year}</Badge>
                        </div>
                        <p className="text-xs text-dark-400">
                          {ref.contribution}
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.section>

        {/* CTA */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <Card hover={false} className="text-center p-12 gradient-border">
            <Shield className="w-16 h-16 text-primary-400 mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-dark-100 mb-4">
              Ready to Try It?
            </h2>
            <p className="text-dark-400 mb-8 max-w-xl mx-auto">
              Test our vulnerability detection system with your own Solidity contracts
              or explore our sample vulnerable contracts.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/analyze">
                <Button size="lg" icon={Zap}>
                  Start Analysis
                </Button>
              </Link>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                <Button variant="secondary" size="lg" icon={Github}>
                  View on GitHub
                </Button>
              </a>
            </div>
          </Card>
        </motion.section>
      </div>
    </div>
  )
}
