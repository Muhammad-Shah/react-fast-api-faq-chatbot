import { createChatBotMessage } from "react-chatbot-kit";
const botName = "Virtural Assistant";
export const config = {
  initialMessages: [createChatBotMessage(`Hey there! I'm ${botName}`)],
  botName: botName,
};
