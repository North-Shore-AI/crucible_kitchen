defmodule CrucibleKitchen.TestSupport.DistillationSamplingClient do
  @moduledoc false

  @behaviour CrucibleTrain.Ports.SamplingClient

  alias CrucibleTrain.Types.ModelInput

  def start_session(_opts, config), do: {:ok, %{id: :teacher, config: config}}

  def sample(_opts, _session, %ModelInput{} = model_input, params, _opts_kw) do
    notify({:teacher_sample, model_input, params})
    {:ok, {:sample, model_input}}
  end

  def sample_stream(_opts, _session, _model_input, _params, _opts_kw),
    do: {:error, :not_implemented}

  def compute_logprobs(_opts, _session, _model_input, _opts_kw),
    do: {:error, :not_implemented}

  def await(_opts, {:sample, %ModelInput{} = model_input}) do
    tokens = ModelInput.all_tokens(model_input)
    response_tokens = tokens |> Enum.take(2) |> Enum.map(&(&1 + 1))

    response = %{
      sequences: [
        %{
          tokens: response_tokens,
          text: "teacher",
          logprobs: Enum.map(response_tokens, fn _ -> -0.1 end),
          stop_reason: "end_of_turn"
        }
      ]
    }

    {:ok, response}
  end

  def close_session(_opts, _session), do: :ok

  defp notify(message) do
    if pid = Process.get(:distill_test_pid) do
      send(pid, message)
    end
  end
end

defmodule CrucibleKitchen.TestSupport.DistillationTrainingClient do
  @moduledoc false

  @behaviour CrucibleTrain.Ports.TrainingClient

  def start_session(_opts, config), do: {:ok, %{id: :distill, config: config}}

  def forward_backward(_opts, _session, datums, opts_kw) do
    notify({:distill_forward_backward, length(datums), opts_kw})
    %{type: :distill_fb, batch_size: length(datums)}
  end

  def forward_backward_custom(_opts, _session, _datums, _loss_fn, _opts_kw),
    do: {:error, :not_implemented}

  def optim_step(_opts, _session, _learning_rate), do: {:error, :not_implemented}

  def await(_opts, %{type: :distill_fb, batch_size: batch_size}) do
    {:ok,
     %{
       metrics: %{
         "loss" => 0.5,
         "num_sequences" => batch_size
       }
     }}
  end

  def save_checkpoint(_opts, _session, _path), do: :ok
  def load_checkpoint(_opts, _session, _path), do: :ok
  def close_session(_opts, _session), do: :ok

  defp notify(message) do
    if pid = Process.get(:distill_test_pid) do
      send(pid, message)
    end
  end
end
