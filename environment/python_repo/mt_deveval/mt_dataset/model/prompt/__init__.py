from typing import List,Dict,Optional

def generate_messages(
    prompt: str,
    system_message: Optional[str],
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    if system_message is None:
        messages = []
    else:
        messages = [
            {"role": "system", "content": system_message}
        ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages