# Third-party models: read before you enable a backend

This repo ships **no model weights**. When a worker boots it downloads the weights for each enabled backend
from Hugging Face (or your S3 cache), and you use them under **that model's license**, not under this repo's
Apache-2.0. You, the deployer, are responsible for complying with it.

| Backend | Weights | License | Can a US / EU enterprise use it? |
|---|---|---|---|
| `wan2.2` (default) | [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B), [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B), ComfyUI repack [Comfy-Org/Wan-2.2](https://huggingface.co/Comfy-Org/Wan-2.2), lightx2v 4-step LoRAs | Apache-2.0 | Yes. Follow the model card's use guidance. |
| `h3` (opt-in) | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), community fine-tune [WarmBloodAban/Minimax-h3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) (a derivative, so still bound by the H3 license) | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | **Not without MiniMax's written license.** |

## Using the `h3` backend: what you need to do

The MiniMax H3 Community License (read the full text; this is a summary, not legal advice):

1. **Territory.** The license covers use "worldwide, excluding" the **United States, the European Union, the
   United Kingdom and the Republic of Korea**. Using, hosting, or displaying H3 or its outputs there is not
   licensed. If you are in one of those places, request a license from MiniMax first: `api@minimax.io`.
2. **Revenue.** If your commercial products and services make **more than USD 20M a year**, you need MiniMax's
   prior written authorization (email `api@minimax.io`, subject "MiniMax H3 licensing - authorization request").
3. **Attribution.** A commercial product or service that uses H3 must prominently show **"MiniMax H3"** in its
   UI, and products built on it must say **"Powered by MiniMax H3"**. Every h3 job result from this MCP carries
   that notice (`notice` field, mp4 `copyright` tag, provenance manifest) so your UI can show it.
4. **Use policy.** Follow the license's acceptable-use exhibit (for example: no military use; disclose AI
   content you publish; outputs may not be used to train other models). The stack's optional prompt guardrail and
   output moderation (both off by default) can help, but they are not a substitute for your own policy and review.
5. **Hosted services.** If you offer it to third parties, bind your users to the same use restrictions and keep
   an abuse reporting channel.

Once you have the license you need, turn it on:

```bash
npx cdk deploy -c media:backends=wan2.2,h3
```

## Adding a backend
A backend is one Python module in `worker/backends/` (weights list + ComfyUI graph builder), one entry in
`lambda/tools/backends.json` (limits the tools Lambda enforces), and its name in `KNOWN_BACKENDS` in the stack.
Add a row to the table above with the license, and call out any territory or revenue limits the way `h3` does.
