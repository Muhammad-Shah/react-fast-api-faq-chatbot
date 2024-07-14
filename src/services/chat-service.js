// import { CHAT_CONSTANTS } from "../config/apiUrls";
// import axios from "../config/axios";
import axios from "axios";

export async function getPromptResponse({ query, onSuccess, onError }) {
  try {
    const response = await axios.get(
      "http://127.0.0.1:8000/api/prompts" + "?" + query
    );
    onSuccess(response?.data?.response);
  } catch (error) {
    onError(error);
  }
}
