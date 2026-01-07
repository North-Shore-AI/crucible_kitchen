defmodule CrucibleKitchen.Stages.DPOForwardBackward do
  @moduledoc """
  DPO forward-backward pass with beta-scaled preference loss.

  Implements the DPO loss function:

      L = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected)))

  This loss encourages the policy to prefer chosen responses over rejected ones,
  with the reference model providing a KL constraint via the log probability differences.

  ## Context Requirements

  **Input:**
  - State: `:preference_batch` - Current batch of preference pairs
  - State: `:ref_chosen_logprobs` - Reference model logprobs for chosen
  - State: `:ref_rejected_logprobs` - Reference model logprobs for rejected
  - State: `:session` - Training session
  - Config: `:dpo_beta` - DPO beta parameter (default: 0.1)

  **Output:**
  - State: `:dpo_future` - Async future for the DPO computation
  - State: `:dpo_loss` - DPO loss value (after await)
  - State: `:dpo_metrics` - DPO-specific metrics

  ## Example

      stage(:dpo_forward_backward, DPOForwardBackward)
      stage(:await_dpo, AwaitFuture, key: :dpo_future)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.TrainingClient
  alias CrucibleTrain.Types.TensorData

  require Logger

  @impl true
  def name, do: :dpo_forward_backward

  @impl true
  def execute(context) do
    session = Context.get_state(context, :session)
    datums = Context.get_state(context, :preference_datums)
    ref_chosen = Context.get_state(context, :ref_chosen_logprobs)
    ref_rejected = Context.get_state(context, :ref_rejected_logprobs)
    beta = Context.get_config(context, :dpo_beta, 0.1)

    run_dpo(context, session, datums, ref_chosen, ref_rejected, beta)
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :preference_datums) == nil ->
        {:error, "preference_datums is required in state (run ComputeReferenceLogprobs first)"}

      Context.get_state(context, :ref_chosen_logprobs) == nil ->
        {:error, "ref_chosen_logprobs is required (run ComputeReferenceLogprobs first)"}

      Context.get_state(context, :ref_rejected_logprobs) == nil ->
        {:error, "ref_rejected_logprobs is required (run ComputeReferenceLogprobs first)"}

      true ->
        :ok
    end
  end

  defp run_dpo(context, session, datums, ref_chosen, ref_rejected, beta) do
    Logger.debug("Running DPO forward-backward with beta=#{beta}")

    ports = get_train_ports(context)
    ref_logprobs = interleave_logprobs(ref_chosen, ref_rejected)

    loss_fn = build_dpo_loss_fn(ref_logprobs, beta)

    future = TrainingClient.forward_backward_custom(ports, session, datums, loss_fn, [])

    context
    |> Context.put_state(:dpo_future, future)
    |> then(&{:ok, &1})
  end

  defp build_dpo_loss_fn(ref_logprobs, beta) do
    ref_logprobs = normalize_logprobs(ref_logprobs)

    fn datums, logprobs_list ->
      weights_list = Enum.map(datums, &weights_to_nx/1)

      chosen = Enum.take_every(logprobs_list, 2)
      rejected = logprobs_list |> Enum.drop(1) |> Enum.take_every(2)
      ref_chosen = Enum.take_every(ref_logprobs, 2)
      ref_rejected = ref_logprobs |> Enum.drop(1) |> Enum.take_every(2)

      chosen_logprob = weighted_logprobs(chosen, weights_list, 0)
      rejected_logprob = weighted_logprobs(rejected, weights_list, 1)
      ref_chosen_logprob = weighted_logprobs(ref_chosen, weights_list, 0)
      ref_rejected_logprob = weighted_logprobs(ref_rejected, weights_list, 1)

      chosen_log_ratio = Nx.subtract(chosen_logprob, ref_chosen_logprob)
      rejected_log_ratio = Nx.subtract(rejected_logprob, ref_rejected_logprob)

      logits = Nx.multiply(beta, Nx.subtract(chosen_log_ratio, rejected_log_ratio))
      losses = Nx.negate(Nx.log(Nx.sigmoid(logits)))
      loss = Nx.mean(losses)

      accuracy =
        chosen_log_ratio
        |> Nx.greater(rejected_log_ratio)
        |> Nx.as_type({:f, 32})
        |> Nx.mean()

      chosen_reward = Nx.multiply(beta, chosen_log_ratio)
      rejected_reward = Nx.multiply(beta, rejected_log_ratio)
      margin = Nx.subtract(chosen_reward, rejected_reward) |> Nx.mean()

      metrics = %{
        loss: Nx.to_number(loss),
        accuracy: Nx.to_number(accuracy),
        chosen_reward: Nx.to_number(Nx.mean(chosen_reward)),
        rejected_reward: Nx.to_number(Nx.mean(rejected_reward)),
        margin: Nx.to_number(margin),
        beta: beta,
        num_pairs: div(length(datums), 2)
      }

      {loss, metrics}
    end
  end

  defp weighted_logprobs(logprobs_list, weights_list, offset) do
    logprobs_list
    |> Enum.with_index()
    |> Enum.map(fn {logprobs, idx} ->
      weights = Enum.at(weights_list, idx * 2 + offset)
      Nx.sum(Nx.multiply(logprobs, weights))
    end)
    |> Nx.stack()
  end

  defp weights_to_nx(datum) do
    datum.loss_fn_inputs["weights"]
    |> TensorData.to_list()
    |> Nx.tensor(type: {:f, 32})
  end

  defp normalize_logprobs(list) do
    Enum.map(list, fn
      %Nx.Tensor{} = tensor -> tensor
      values when is_list(values) -> Nx.tensor(values, type: {:f, 32})
      other -> Nx.tensor(other, type: {:f, 32})
    end)
  end

  defp interleave_logprobs(chosen, rejected) do
    Enum.zip(chosen, rejected)
    |> Enum.flat_map(fn {c, r} -> [c, r] end)
  end
end
