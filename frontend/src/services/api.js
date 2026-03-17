const API_BASE_URL = '/api'

/**
 * Analyze a smart contract for vulnerabilities
 * @param {string} code - Solidity contract source code
 * @returns {Promise<Object>} Analysis results
 */
export async function analyzeContract(code) {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ code }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || 'Analysis failed')
  }

  return response.json()
}

/**
 * Get sample contracts for demo
 * @returns {Promise<Object>} List of sample contracts
 */
export async function getSampleContracts() {
  const response = await fetch(`${API_BASE_URL}/sample-contracts`)

  if (!response.ok) {
    throw new Error('Failed to fetch sample contracts')
  }

  return response.json()
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`)

  if (!response.ok) {
    throw new Error('API is not healthy')
  }

  return response.json()
}

/**
 * Get vulnerability class information
 * @returns {Promise<Object>} Vulnerability classes
 */
export async function getVulnerabilityClasses() {
  const response = await fetch(`${API_BASE_URL}/vulnerability-classes`)

  if (!response.ok) {
    throw new Error('Failed to fetch vulnerability classes')
  }

  return response.json()
}

/**
 * Get AI-powered explanation and fix for a vulnerability
 * @param {string} code - Solidity contract source code
 * @param {string} issue - Vulnerability type: reentrancy, access_control, arithmetic, unchecked_calls
 * @returns {Promise<Object>} AI response with explanation and fix
 */
export async function getAIHelp(code, issue) {
  const response = await fetch(`${API_BASE_URL}/ai-help`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, issue }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || 'AI help request failed')
  }

  return response.json()
}

/**
 * Get model architecture information
 * @returns {Promise<Object>} Model info
 */
export async function getModelInfo() {
  const response = await fetch(`${API_BASE_URL}/model-info`)
  if (!response.ok) throw new Error('Failed to fetch model info')
  return response.json()
}

/**
 * Send a chat message to the Gemini chatbot
 * @param {Array} messages - [{role, content}]
 * @param {string} contractContext - Solidity source
 * @param {string} analysisContext - Analysis summary text
 */
export async function sendChat(messages, contractContext = '', analysisContext = '') {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, contract_context: contractContext, analysis_context: analysisContext }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || 'Chat failed')
  }
  return response.json()
}

/**
 * Run Slither static analysis on contract code
 * @param {string} code - Solidity source
 */
export async function runSlither(code) {
  const response = await fetch(`${API_BASE_URL}/slither`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || 'Slither failed')
  }
  return response.json()
}

/**
 * Run Mythril SWC analysis on contract code
 * @param {string} code - Solidity source
 */
export async function runMythril(code) {
  const response = await fetch(`${API_BASE_URL}/mythril`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || 'Mythril failed')
  }
  return response.json()
}
