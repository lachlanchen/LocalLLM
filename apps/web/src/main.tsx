import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { schedulePwaServiceWorkerRegistration } from './pwa'
import './styles.css'

schedulePwaServiceWorkerRegistration(
  window,
  'serviceWorker' in navigator ? navigator.serviceWorker : undefined,
)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
