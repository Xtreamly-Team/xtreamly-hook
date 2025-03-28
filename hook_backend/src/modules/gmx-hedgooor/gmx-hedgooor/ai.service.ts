import {get} from "@app/utils/api";

export function getPolicy(userId: string): any {
  return get(process.env.AI_BACKEND_URL!, "hedge", {
    "user_id": userId,
  })
}