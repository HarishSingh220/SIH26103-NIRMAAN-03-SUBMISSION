import PlaceholderPage from "./PlaceholderPage";
import { AIIcon } from "../components/Icons";

function AIAssistant() {
  return (
    <PlaceholderPage
      eyebrow="IPMD • AI ASSISTANT"
      title="AI Assistant"
      subtitle="A conversational assistant for querying project status, cost overruns and risk trends will be built out on this page."
      icon={<AIIcon />}
    />
  );
}

export default AIAssistant;
