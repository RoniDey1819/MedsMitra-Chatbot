// MedsMitra - shared front-end behaviour (no backend; everything is simulated in-memory)

document.addEventListener("DOMContentLoaded", function () {
  /* ---------- Cart badge (simulated, persists only for this page view) ---------- */
  var cartCountEls = document.querySelectorAll(".cart-count");
  var cartCount = 3;
  function paintCartCount() {
    cartCountEls.forEach(function (el) {
      el.textContent = cartCount;
    });
  }
  paintCartCount();

  /* ---------- Toast helper ---------- */
  var toast = document.createElement("div");
  toast.className = "rx-toast";
  toast.innerHTML =
    '<i class="fa-solid fa-circle-check"></i><span id="rxToastMsg">Added to cart</span>';
  document.body.appendChild(toast);
  var toastMsg = toast.querySelector("#rxToastMsg");
  var toastTimer;
  function showToast(msg) {
    toastMsg.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toast.classList.remove("show");
    }, 2200);
  }

  /* ---------- Add-to-cart buttons ---------- */
  document.querySelectorAll("[data-add-cart]").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      cartCount += 1;
      paintCartCount();
      var name = btn.getAttribute("data-add-cart") || "Item";
      showToast(name + " added to cart");
    });
  });

  /* ---------- FAQ accordion (custom, accessible) ---------- */
  document.querySelectorAll(".faq-q").forEach(function (q) {
    q.addEventListener("click", function () {
      var expanded = q.getAttribute("aria-expanded") === "true";
      var answer = document.getElementById(q.getAttribute("aria-controls"));
      q.setAttribute("aria-expanded", String(!expanded));
      if (answer) answer.style.display = expanded ? "none" : "block";
    });
  });

  /* ---------- Back to top ---------- */
  var backTop = document.querySelector(".back-to-top");
  if (backTop) {
    window.addEventListener("scroll", function () {
      backTop.classList.toggle("show", window.scrollY > 500);
    });
    backTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* ---------- Newsletter / contact forms: prevent real submit, show confirmation ---------- */
  document.querySelectorAll("form[data-demo-form]").forEach(function (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      showToast(
        form.getAttribute("data-demo-form") || "Submitted - thank you!",
      );
      form.reset();
    });
  });

  /* ---------- Mark active nav link ---------- */
  var path = window.location.pathname.split("/").pop() || "index.html";
  document
    .querySelectorAll(".main-nav .nav-link[href]")
    .forEach(function (link) {
      if (link.getAttribute("href") === path) link.classList.add("active");
    });
});
