defmodule CrucibleKitchen.TestSupport.DPOTestSamplingClient do
  @moduledoc false

  @behaviour CrucibleTrain.Ports.SamplingClient

  alias CrucibleTrain.Types.ModelInput

  def start_session(_opts, config), do: {:ok, %{id: :sampling, config: config}}

  def sample(_opts, _session, _model_input, _params, _opts_kw), do: {:error, :not_implemented}

  def sample_stream(_opts, _session, _model_input, _params, _opts_kw),
    do: {:error, :not_implemented}

  def compute_logprobs(_opts, _session, model_input, _opts_kw) do
    notify({:compute_logprobs, model_input})
    {:ok, {:logprobs, model_input}}
  end

  def await(_opts, {:logprobs, %ModelInput{} = model_input}) do
    tokens = ModelInput.all_tokens(model_input)
    tail = tokens |> Enum.drop(1) |> Enum.map(&(-1.0 * &1))
    {:ok, [nil | tail]}
  end

  def close_session(_opts, _session), do: :ok

  defp notify(message) do
    if pid = Process.get(:dpo_test_pid) do
      send(pid, message)
    end
  end
end

defmodule CrucibleKitchen.TestSupport.DPOTestTrainingClient do
  @moduledoc false

  @behaviour CrucibleTrain.Ports.TrainingClient

  alias CrucibleTrain.Types.TensorData

  def start_session(_opts, config), do: {:ok, %{id: :training, config: config}}

  def forward_backward(_opts, _session, _datums, _opts_kw), do: {:error, :not_implemented}

  def forward_backward_custom(_opts, _session, datums, loss_fn, _opts_kw) do
    notify({:forward_backward_custom, length(datums)})

    logprobs_list =
      Enum.with_index(datums)
      |> Enum.map(fn {datum, idx} ->
        length =
          datum.loss_fn_inputs["target_tokens"]
          |> TensorData.to_list()
          |> length()

        values = Enum.map(1..length, fn step -> -0.1 * (idx + 1) * step end)
        Nx.tensor(values)
      end)

    {loss, metrics} = loss_fn.(datums, logprobs_list)

    %{type: :dpo_future, loss: loss, metrics: metrics}
  end

  def optim_step(_opts, _session, _learning_rate), do: {:error, :not_implemented}

  def await(_opts, %{type: :dpo_future, loss: loss, metrics: metrics}) do
    {:ok, %{loss: Nx.to_number(loss), metrics: metrics}}
  end

  def save_checkpoint(_opts, _session, _path), do: :ok
  def load_checkpoint(_opts, _session, _path), do: :ok
  def close_session(_opts, _session), do: :ok

  defp notify(message) do
    if pid = Process.get(:dpo_test_pid) do
      send(pid, message)
    end
  end
end
