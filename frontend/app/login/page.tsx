import { AuthForm } from "@/components/AuthForm";

export default function LoginPage() {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <AuthForm mode="login" />
    </div>
  );
}
