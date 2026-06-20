# 安全：Secrets、API Key Rotation、Audit Logs、Guardrails

> 通过 centralized vaults（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除 secret sprawl。永远不要把 credentials 存在 config files、进入 VCS 的 env files、spreadsheets 里。优先使用 IAM roles，而不是 static keys；CI/CD 使用 OIDC。AI-gateway pattern 是 2026 年的解决方案：apps → gateway → model provider，gateway 在运行时从 vault 拉 credentials。在 vault 里 rotate，所有 apps 会在几分钟内拿到新 key，不需要 redeploy，也不需要 Slack 里问“谁有新 key”。Rotation policy ≤90 天；每个 commit 都用 TruffleHog / GitGuardian / Gitleaks 扫描。Zero-trust：MFA、SSO、RBAC/ABAC、short-lived tokens、device posture。PII scrubbing 用 entity recognition 在转发前 mask PHI/PII；consistent tokenization（Mesh approach）把 sensitive values 映射到稳定 placeholders，这样 LLM 仍能保留 code/relationship semantics。Network egress：LLM services 放在专用 VPC/VNet subnet，只 whitelist `api.openai.com`、`api.anthropic.com` 等；阻断所有其他 outbound。2026 年的 incident driver：Vercel supply-chain attack 通过 compromised CI/CD credentials 窃取了数千个 customer deployments 的 env vars。

**Type:** Learn
**Languages:** Python（stdlib，toy PII-scrubber + audit-log writer）
**Prerequisites:** Phase 17 · 19（AI Gateways），Phase 17 · 13（Observability）
**Time:** ~60 minutes

## 学习目标

- 枚举四个 secret-management anti-patterns（VCS 中的 config files、hardcoded env、spreadsheets、static keys），并说出对应替代方案。
- 解释 AI-gateway-pulls-from-vault pattern 为什么是 2026 production standard。
- 实现带 consistent tokenization 的 PII scrubber（same value → same placeholder），让语义关系保留下来。
- 说出 2026 Vercel supply-chain incident，以及它对 CI/CD credential hygiene 的教训。

## 问题

实习生提交了带 API keys 的 `.env`。他们很快删掉了。问题是 keys 已经进入 git history。GitGuardian scan 抓到了它，而你的 rotation process 是“Slack 全组、更新 40 个 config files、重新部署所有 services”。8 小时后，一半 services 已上线，一半还在等 deploy window。

与此同时，user prompts 里包含：“My SSN is 123-45-6789.” Prompt 被发送给 OpenAI。你有 BAA，但内部 policy 要求在转发前 mask PII。你没有做。

再与此同时，你的 EKS cluster 中的 LLM pod 可以访问任意 internet host。有人通过 DNS lookup 把数据 exfil 到 attacker-controlled domain。没有任何东西挡住它。

LLM services 的安全必须同时处理这三个 vectors：vault-backed credentials、PII scrubbing、network egress filtering、audit logs。

## 概念

### Centralized vault + IAM-role pull

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一 truth source。

**IAM role**：app/gateway 通过自己的 IAM identity 认证，而不是使用 static key。Vault 在 token lifetime 内返回 secret。

**AI-gateway pattern**：gateway 在 request time 从 vault 拉取 `OPENAI_API_KEY`。在 vault 中 rotate；下一个 request 就拿到新 key。不需要 redeploy。

### Rotation policy ≤ 90 天

覆盖所有 API keys、vault root tokens、CI/CD credentials。能自动 rotation 的就自动化。Manual rotation 要被记录和追踪。

### Secret scanning

- **TruffleHog**：对 commits 做 regex + entropy 检测。
- **GitGuardian**：commercial，高准确率。
- **Gitleaks**：OSS，在 CI 中运行。

每个 commit 都运行。检测到新 secret 时阻断 PR。

### Zero-trust posture

- 所有 accounts 都要求 MFA。
- 通过 SAML/OIDC 做 SSO。
- RBAC（role-based）或 ABAC（attribute-based）用于 fine-grained access。
- Short-lived tokens（以小时计，而不是以天计）。
- Device posture：只允许带 disk encryption 的 corp devices。

### PII / PHI scrubbing

在 prompt 离开你的 infra 之前：

1. Entity recognition（spaCy NER、Presidio、commercial）。
2. Mask matched entities：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. Consistent tokenization（Mesh approach）：同一个 value 映射到同一个 placeholder，让 LLM 保留关系。
4. 可选 reverse mapping，用于 LLM response。

Static regex filters 能抓住基础 patterns；NER 能抓更多。两者都用。

### Input + output guardrails

Input：阻断已知 jailbreaks、forbidden topics；按 user 做 rate-limit。

Output：用 regex scrub leaked secrets（API key patterns、refusal contexts 中的 email patterns），用 classifier 检测 policy violations。

### Network egress whitelist

LLM services 放在专用 subnet：

- Whitelist：`api.openai.com`、`api.anthropic.com`、vector DB endpoints、vault endpoints。
- 其他全部 drop。
- DNS 走 allowlist-only resolver（避免 DNS-tunneling exfil）。

### Audit log

每次 LLM call 都写入 immutable log，包含：

- Timestamp。
- User / tenant。
- Prompt hash（为了 privacy，不记录 raw prompt）。
- Model + version。
- Token counts。
- Cost。
- Response hash。
- 任何 guardrail trips。

按 regulatory requirement 保留（SOC 2 1 年，HIPAA 6 年）。

### 2026 Vercel incident

Supply-chain attack：compromised CI/CD credentials 窃取了数千个 customer deployments 的 env vars。教训：CI/CD credentials 等同于 production credentials。存进 vault。缩小 scope。积极 rotate。

### 你应该记住的数字

- Rotation policy：≤90 天。
- 每个 commit 都扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD creds compromised → 数千个 customer env vars 泄露。
- Audit log retention：SOC 2 = 1 年，HIPAA = 6 年。

## Use It

`code/main.py` 实现了一个 toy PII scrubber，带 consistent tokenization 和 append-only audit log。

## Ship It

本课产出 `outputs/skill-llm-security-plan.md`。给定 regulatory scope 和 current state，它会规划 vault migration、scrubber、egress、audit log。

## 练习

1. 运行 `code/main.py`。发送两个引用同一个 SSN 的 prompts。确认两次都得到同一个 placeholder。
2. 为一个在 EKS 上运行 vLLM、同时调用 OpenAI + Anthropic + Weaviate 的 deployment 设计 network egress policy。
3. 你在 git history 中发现了一个 key（2 年前的）。正确响应是什么：rotate key、scrub history，还是两者都做？说明理由。
4. 你的 audit log 每天增长 10 GB。设计 retention tiers（hot 30d、warm 12mo、cold 6yr）。
5. 论证 reverse-tokenization（把真实值替换回 LLM response）是否值得增加复杂度，还是应该让 placeholders 保持可见。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Vault | “secrets store” | 集中的 credential management service。 |
| IAM role | “identity-based auth” | App assume 的 role；返回 short-lived creds。 |
| OIDC for CI/CD | “cloud-issued tokens” | CI 中没有 static keys；通过 OIDC 表达 identity。 |
| TruffleHog / GitGuardian / Gitleaks | “secret scanners” | Commit-time secret detection。 |
| RBAC / ABAC | “access control” | Role-based vs attribute-based。 |
| PII scrubbing | “data masking” | 移除或 tokenize sensitive entities。 |
| Consistent tokenization | “stable placeholders” | 同一个 value 每次映射到同一个 token。 |
| Mesh approach | “Mesh tokenization” | 保留语义关系的 tokenization pattern。 |
| Egress whitelist | “outbound allowlist” | 只有被允许的 domains 可访问。 |
| Audit log | “immutable history” | 用于 compliance 的 append-only record。 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
