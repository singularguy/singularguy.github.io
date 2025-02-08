(() => {
  // <stdin>
  function toggleTheme() {
    if (document.body.className.includes("dark")) {
      document.body.classList.remove("dark");
      localStorage.setItem("theme", "light");
    } else {
      document.body.classList.add("dark");
      localStorage.setItem("theme", "dark");
    }
  }
  window.addEventListener("DOMContentLoaded", (event) => {
    const switcher = document.getElementById("theme-switcher");
    if (switcher) {
      switcher.addEventListener("click", () => {
        toggleTheme();
      });
    }
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (event2) => {
      if (event2.matches && localStorage.getItem("theme") === "light") {
        toggleTheme();
      }
    });
  });
})();
