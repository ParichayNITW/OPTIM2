import React, { createContext, useCallback, useEffect, useState } from 'react'
import { getMe, login as apiLogin, register as apiRegister } from '../api/auth'
import type { User } from '../types'

interface AuthContextValue {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, name?: string) => Promise<void>
  logout: () => void
}

export const AuthContext = createContext<AuthContextValue>({} as AuthContextValue)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('access_token')
    if (!token) { setIsLoading(false); return }
    getMe()
      .then(setUser)
      .catch(() => localStorage.removeItem('access_token'))
      .finally(() => setIsLoading(false))
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const { access_token } = await apiLogin(email, password)
    localStorage.setItem('access_token', access_token)
    const me = await getMe()
    setUser(me)
  }, [])

  const register = useCallback(async (email: string, password: string, name = '') => {
    const { access_token } = await apiRegister(email, password, name)
    localStorage.setItem('access_token', access_token)
    const me = await getMe()
    setUser(me)
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('access_token')
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}
