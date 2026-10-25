import { useState } from 'react';
import { RouterProvider } from 'react-router-dom';
import { Providers } from './providers';
import { createAppRouter } from './router';

export function App() {
  // Created once per mount so the router is stable across re-renders.
  const [router] = useState(createAppRouter);
  return (
    <Providers>
      <RouterProvider router={router} />
    </Providers>
  );
}
