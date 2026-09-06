# Public Honcho turn-path harness

Production path under test:

1. The principal tests call `agent.conversation_loop.run_conversation(host, query, task_id="task-public-harness")` directly. The host is a production `run_agent.AIAgent`, initialized offline with context files, built-in memory initialization, and background review disabled because those host-shell services are unrelated to Honcho recall.
2. `run_conversation` calls the real `agent.turn_context.build_turn_context` prologue exactly as production does.
3. `build_turn_context` increments the host user-turn count, then calls the real `MemoryManager.on_turn_start(turn_number, original_user_message)` before retrieval.
4. `MemoryManager.on_turn_start` calls the real `HonchoMemoryProvider.on_turn_start`, which creates one immutable `ProviderTurnContext` for the initialized workspace, session, user peer, assistant peer, absolute monotonic deadline, and shared ledger.
5. The same prologue calls the real `MemoryManager.prefetch_all(original_user_message)`. For the external provider, `MemoryManager._prefetch_provider` starts and joins the production `memory-prefetch-honcho` worker with copied context variables.
6. `HonchoMemoryProvider.prefetch` uses its production lifecycle and owned-worker scheduling for first-turn base retrieval and dialectic work. Nested `HonchoSessionManager` calls inherit the same turn context. Real `honcho-ai==2.2.0` objects serialize requests and deserialize responses. The accounting transport binds every scoped SDK request to the same workspace authority, absolute deadline, and transport ledger before forwarding to `httpx.MockTransport`.
7. `build_turn_context` composes external memory into the current user row. `run_conversation` then performs its real outgoing-message projection, selection, sidecar substitution, sanitization, normalization, system-message insertion, API-kwargs construction, and request middleware.
8. The capture seam is `CaptureModelClient.create(**kwargs)`, the exact Chat Completions model/provider boundary reached by the production non-streaming request path. Tests inspect `kwargs["messages"]`; they do not use `TurnContext.api_content` as proof. The fake returns one deterministic SDK-shaped terminal assistant response so the real loop and finalizer complete.
9. Teardown calls `MemoryManager.shutdown_all`, which drains its executor and invokes the real Honcho provider/session/SDK shutdown path under the existing deadline rules.

Mock and inert-seam inventory:

- `httpx.MockTransport(PublicTurnTransport)` is the only substantive Honcho/provider-network mock. It returns SDK-shaped workspace, peer, session, context, search, dialectic, conclusion, and message-write responses.
- `CaptureModelClient.create` is the permitted deterministic fake at the exact model/API boundary and records the final API request.
- `run_agent.OpenAI` construction is redirected to that capture client so both the shared and request-local production client paths stay offline.
- `run_agent.get_tool_definitions` returns an empty registry and `run_agent.check_toolset_requirements` returns an empty result. These are inert unrelated host-shell services; this recall harness exercises no model tools.
- A temporary `config.yaml` supplies the synthetic model context length, preventing unrelated model-metadata network probing.
- `AIAgent(skip_context_files=True, skip_memory=True, skip_background_review=True)` suppresses unrelated host startup services. Immediately after construction, `_memory_manager` is set to the real initialized `MemoryManager` containing the real `HonchoMemoryProvider`.

Live, unmocked seams:

- `run_conversation`
- `build_turn_context`
- `MemoryManager.on_turn_start`
- `MemoryManager.prefetch_all` and provider scheduling
- `HonchoMemoryProvider` lifecycle and workers
- real `honcho-ai==2.2.0`
- workspace proof
- returned-record scope/source validator
- renderer
- turn ledger
- absolute deadline
- direct retrieval, base context, and dialectic retrieval
- outgoing message selection, sanitization, normalization, and final API request assembly

The Pinewood tests assert final API-bound messages preserve scoped/provenanced direct evidence before generated representation, reject wrong workspace/session/peer/source/entity records, retain both distinct direct concerns under duplicate pressure, hide IDs/timestamps/metadata/raw provider errors, preserve first-turn prewarm/query cadence, and replace turn authority after reset while retaining only the permitted provider recall pool.

Conclusion tool-loop coverage:

- The capture client can return a synthetic `honcho_conclude` list tool call as its first response and a terminal answer as its second response.
- `run_conversation` validates the call and dispatches it through the production `AIAgent._execute_tool_calls` / `_invoke_tool` path into the real `MemoryManager.handle_tool_call` and `HonchoMemoryProvider` implementation.
- The second captured API request proves the model-visible tool result contains a scoped local handle and semantic conclusion text while excluding the provider conclusion ID and timestamp.
