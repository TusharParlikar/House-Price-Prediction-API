// Sends the form to the API and shows the predicted price or the errors.

const form = document.getElementById("house-form");
const apiUrl = document.getElementById("api-url");
const result = document.getElementById("result");
const button = form.querySelector("button");

// Step 1: remember the API URL between visits (localStorage can be blocked, so ignore failures)
try {
  apiUrl.value = localStorage.getItem("apiUrl") || apiUrl.value;
} catch {}
apiUrl.addEventListener("change", () => {
  try { localStorage.setItem("apiUrl", apiUrl.value); } catch {}
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // Step 2: turn the form into the JSON body. Every field is a number except ocean_proximity.
  const house = Object.fromEntries(new FormData(form));
  for (const key in house) {
    if (key !== "ocean_proximity") house[key] = Number(house[key]);
  }

  button.disabled = true;
  result.textContent = "Predicting...";

  try {
    // Step 3: call the API. Strip a trailing "/" so "host/" and "host" both work.
    const res = await fetch(apiUrl.value.replace(/\/+$/, "") + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(house),
    });
    const data = await res.json();

    // Step 4: show the price, or list each field the API rejected (422)
    if (res.ok) {
      const price = data.predicted_price.toLocaleString("en-US", {
        style: "currency", currency: "USD", maximumFractionDigits: 0,
      });
      result.innerHTML = `<div>Predicted median house value</div><div class="price">${price}</div>`;
    } else {
      showErrors(data.detail);
    }
  } catch (err) {
    // Network error: API not running, wrong URL, or a hosted API still waking up
    showErrors([{ loc: ["network"], msg: `Could not reach the API (${err.message}). Is it running at ${apiUrl.value}?` }]);
  } finally {
    button.disabled = false;
  }
});

// Build the error list with textContent, never innerHTML, so API text can't inject HTML
function showErrors(detail) {
  const errors = Array.isArray(detail) ? detail : [{ loc: [], msg: String(detail) }];
  const box = document.createElement("div");
  box.className = "error";
  box.textContent = "Request rejected:";
  const list = document.createElement("ul");
  for (const e of errors) {
    const item = document.createElement("li");
    // loc looks like ["body", "households"]; show just the field name
    const field = (e.loc || []).filter((part) => part !== "body").join(".");
    item.textContent = field ? `${field}: ${e.msg}` : e.msg;
    list.append(item);
  }
  box.append(list);
  result.replaceChildren(box);
}
