import Config

config :exla, :add_backend_on_inspect, config_env() != :test

if System.get_env("USE_EXLA_AT_COMPILE_TIME") do
  config :nx, :default_backend, EXLA.Backend
end
