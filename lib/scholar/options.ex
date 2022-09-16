defmodule Scholar.Options do
  # Useful NimbleOptions validations.
  @moduledoc false

  def positive_number(num) do
    if is_number(num) and num >= 0 do
      {:ok, num}
    else
      {:error, "expected positive number, got: #{inspect(num)}"}
    end
  end
end
