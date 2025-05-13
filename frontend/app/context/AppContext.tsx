import { createContext, useState, ReactNode } from 'react';

// Define the shape of the context
interface AppContextType {
  globalState: string;
  setGlobalState: React.Dispatch<React.SetStateAction<string>>;
}

// Create context with a default value
export const AppContext = createContext<AppContextType | undefined>(undefined);

// AppProvider component to wrap your app
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [globalState, setGlobalState] = useState<string>("This is global state");

  return (
    <AppContext.Provider value={{ globalState, setGlobalState }}>
      {children}
    </AppContext.Provider>
  );
};