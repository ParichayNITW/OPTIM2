import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { usePipelineStore } from '../store/pipelineStore'

const navItems = [
  { to: '/',          label: 'Dashboard',  icon: '⊞' },
  { to: '/optimize',  label: 'Optimize',   icon: '▶' },
  { to: '/history',   label: 'Job History',icon: '☰' },
  { to: '/pipelines', label: 'Pipelines',  icon: '⊜' },
]

export function Layout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const { loadCase, saveCase } = usePipelineStore()

  const handleLogout = () => { logout(); navigate('/login') }

  const handleImport = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        const json = JSON.parse(text)
        loadCase(json)
        navigate('/optimize')
      } catch {
        alert('Invalid JSON file')
      }
    }
    input.click()
  }

  const handleExport = () => {
    const data = saveCase()
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pipeline_case.json'
    a.click()
    URL.revokeObjectURL(url)
  }

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

        <div className="px-3 py-3 border-t border-brand-700 space-y-1.5">
          <button
            onClick={handleImport}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium text-blue-200 hover:bg-brand-700 hover:text-white transition-colors"
          >
            <span>⬆</span> Import JSON
          </button>
          <button
            onClick={handleExport}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium text-blue-200 hover:bg-brand-700 hover:text-white transition-colors"
          >
            <span>⬇</span> Export JSON
          </button>
        </div>

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
        {location.pathname === '/optimize'
          ? <Outlet />
          : <div className="max-w-7xl mx-auto px-6 py-6"><Outlet /></div>
        }
      </main>
    </div>
  )
}
