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
