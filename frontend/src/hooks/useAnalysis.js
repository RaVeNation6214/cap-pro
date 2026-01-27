import { useState, useCallback } from 'react'
import { analyzeContract } from '../services/api'

/**
 * Custom hook for contract analysis
 */
export function useAnalysis() {
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyze = useCallback(async (code) => {
    if (!code?.trim()) {
      setError('Please provide contract code to analyze')
      return null
    }

    setIsLoading(true)
    setError(null)

    try {
      const analysisResult = await analyzeContract(code)
      setResult(analysisResult)
      return analysisResult
    } catch (err) {
      const errorMessage = err.message || 'Analysis failed. Please try again.'
      setError(errorMessage)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
    setIsLoading(false)
  }, [])

  return {
    result,
    isLoading,
    error,
    analyze,
    reset,
  }
}

/**
 * Custom hook for managing analysis history (local storage)
 */
export function useAnalysisHistory() {
  const STORAGE_KEY = 'smart_audit_history'
  const MAX_HISTORY = 10

  const getHistory = useCallback(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  }, [])

  const addToHistory = useCallback((code, result) => {
    const history = getHistory()
    const entry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      codePreview: code.substring(0, 100) + (code.length > 100 ? '...' : ''),
      riskLevel: result.risk_level,
      overallRisk: result.overall_risk_score,
    }

    const updatedHistory = [entry, ...history.slice(0, MAX_HISTORY - 1)]

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedHistory))
    } catch {
      // Storage might be full
    }

    return updatedHistory
  }, [getHistory])

  const clearHistory = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  return {
    getHistory,
    addToHistory,
    clearHistory,
  }
}

export default useAnalysis
