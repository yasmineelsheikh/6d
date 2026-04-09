export default function DemoLayout({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ color: '#111827', backgroundColor: '#f9fafb', minHeight: '100vh' }}>
      {children}
    </div>
  )
}
