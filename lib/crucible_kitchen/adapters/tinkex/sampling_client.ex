defmodule CrucibleKitchen.Adapters.Tinkex.SamplingClient do
  @moduledoc """
  SamplingClient adapter for the Tinker ML platform.
  """

  @behaviour CrucibleTrain.Ports.SamplingClient

  require Logger

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  @type session :: %{
          id: pid(),
          service_client: pid(),
          model: String.t() | nil,
          config: map()
        }

  @impl true
  @spec start_session(keyword(), map()) :: {:ok, session()} | {:error, term()}
  def start_session(adapter_opts, config) do
    api_key = Keyword.get(adapter_opts, :api_key) || System.get_env("TINKER_API_KEY")

    base_url =
      Keyword.get(adapter_opts, :base_url) || System.get_env("TINKER_BASE_URL", @default_base_url)

    if api_key do
      do_start_session(config, api_key, base_url)
    else
      {:error, {:missing_env, "TINKER_API_KEY"}}
    end
  end

  @impl true
  @spec sample(keyword(), session(), map(), map(), keyword()) ::
          {:ok, Task.t()} | {:error, term()}
  def sample(_opts, %{id: sampling_client}, model_input, params, opts_kw) do
    Tinkex.SamplingClient.sample(sampling_client, model_input, params, opts_kw)
  end

  @impl true
  @spec sample_stream(keyword(), session(), map(), map(), keyword()) ::
          {:ok, Enumerable.t()} | {:error, term()}
  def sample_stream(_opts, %{id: sampling_client}, model_input, params, opts_kw) do
    Tinkex.SamplingClient.sample_stream(sampling_client, model_input, params, opts_kw)
  end

  @impl true
  @spec compute_logprobs(keyword(), session(), map(), keyword()) ::
          {:ok, Task.t()} | {:error, term()}
  def compute_logprobs(_opts, %{id: sampling_client}, model_input, opts_kw) do
    Tinkex.SamplingClient.compute_logprobs(sampling_client, model_input, opts_kw)
  end

  @impl true
  @spec await(keyword(), Task.t()) :: {:ok, term()} | {:error, term()}
  def await(_opts, task) do
    Task.await(task, :infinity)
  end

  @impl true
  @spec close_session(keyword(), session()) :: :ok
  def close_session(_opts, %{service_client: service_client}) do
    if Process.alive?(service_client) do
      GenServer.stop(service_client, :normal)
    end

    :ok
  end

  defp do_start_session(config, api_key, base_url) do
    tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)

    with {:ok, service_client} <- create_service_client(tinkex_config),
         {:ok, sampling_client} <- create_sampling_client(service_client, config) do
      {:ok,
       %{
         id: sampling_client,
         service_client: service_client,
         model: resolve_model(config),
         config: config
       }}
    end
  end

  defp create_service_client(tinkex_config) do
    Logger.debug("Creating Tinkex ServiceClient for sampling...")

    case Tinkex.ServiceClient.start_link(config: tinkex_config) do
      {:ok, pid} ->
        Logger.debug("ServiceClient created")
        {:ok, pid}

      {:error, reason} ->
        {:error, {:service_client_failed, reason}}
    end
  end

  defp create_sampling_client(service_client, config) do
    opts = sampling_opts(config)
    Tinkex.ServiceClient.create_sampling_client(service_client, opts)
  end

  defp sampling_opts(config) do
    model_path =
      Map.get(config, :model_path) ||
        Map.get(config, :checkpoint_path) ||
        Map.get(config, "model_path") ||
        Map.get(config, "checkpoint_path")

    base_model =
      Map.get(config, :model) ||
        Map.get(config, :base_model) ||
        Map.get(config, "model") ||
        Map.get(config, "base_model")

    []
    |> maybe_put(:model_path, model_path)
    |> maybe_put(:base_model, base_model)
  end

  defp resolve_model(config) do
    Map.get(config, :model) ||
      Map.get(config, :base_model) ||
      Map.get(config, "model") ||
      Map.get(config, "base_model")
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
