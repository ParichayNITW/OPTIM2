import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

export function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await register(email, password, name)
      navigate('/')
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="text-2xl font-bold text-brand-900">Pipeline Optima™</div>
        </div>
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow p-8 space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">Create account</h2>
          {error && <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded">{error}</div>}
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Password (min 8 chars)</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={8}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500" />
          </div>
          <button type="submit" disabled={loading}
            className="w-full bg-brand-600 hover:bg-brand-700 text-white rounded-md py-2 text-sm font-medium transition-colors disabled:opacity-60">
            {loading ? 'Creating…' : 'Create account'}
          </button>
          <p className="text-center text-xs text-gray-500">
            Already have an account? <Link to="/login" className="text-brand-600 hover:underline">Sign in</Link>
          </p>
        </form>
      </div>
    </div>
  )
}
