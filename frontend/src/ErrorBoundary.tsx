import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("BizCheck UI failed to render", error, info);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <main className="fatal-error">
          <AlertTriangle aria-hidden="true" size={28} />
          <h1>頁面暫時無法顯示</h1>
          <p>重新整理後即可再次查詢，不會影響政府公開資料。</p>
          <button type="button" onClick={() => window.location.reload()}>
            <RotateCcw aria-hidden="true" size={18} />
            重新整理
          </button>
        </main>
      );
    }

    return this.props.children;
  }
}
