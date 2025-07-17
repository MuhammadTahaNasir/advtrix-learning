export function stripUnwantedAttributes() {
  const body = document.querySelector('body');
  if (body) {
    body.removeAttribute('data-new-gr-c-s-check-loaded');
    body.removeAttribute('data-gr-ext-installed');
  }
}