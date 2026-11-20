import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

const navItems = [
  { to: '/',          label: 'Dashboard',  icon: '⊞' },
  { to: '/optimize',  label: 'Optimize',   icon: '▶' },
  { to: '/history',   label: 'Job History',icon: '☰' },
  { to: '/pipelines', label: 'Pipelines',  icon: '⊜' },
]

export function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const handleLogout = () => { logout(); navigate('/login') }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-56 bg-brand-900 text-white flex flex-col shrink-0">
        <div className="px-5 py-5 border-b border-brand-700">
          <div className="text-lg font-bold tracking-tight">Pipeline Optima™</div>
          <div className="text-xs text-blue-300 mt-0.5">Globally optimal scheduling</div>
        </div>

        <nav className="flex-1 py-4 space-y-1 px-2">
          {navItems.map(({ to, label, icon }) => {
            const active = location.pathname === to
            return (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  active ? 'bg-brand-700 text-white' : 'text-blue-200 hover:bg-brand-700 hover:text-white'
                }`}
              >
                <span>{icon}</span>
                {label}
              </Link>
            )
          })}
        </nav>

        <div className="px-4 py-4 border-t border-brand-700">
          <div className="text-xs text-blue-300 truncate">{user?.email}</div>
          <button
            onClick={handleLogout}
            className="mt-2 text-xs text-blue-300 hover:text-white transition-colors"
          >
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-6 py-6">{children}</div>
      </main>
    </div>
  )
}
