import client from './client'
import type { TokenResponse, User } from '../types'

export const register = (email: string, password: string, name = '') =>
  client.post<TokenResponse>('/auth/register', { email, password, name }).then(r => r.data)

export const login = (email: string, password: string) =>
  client.post<TokenResponse>('/auth/login', { email, password }).then(r => r.data)

export const getMe = () =>
  client.get<User>('/auth/me').then(r => r.data)
