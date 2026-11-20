import { Navigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth()
  if (isLoading) return <div className="flex items-center justify-center h-screen text-gray-500">Loading…</div>
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}
