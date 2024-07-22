import { html } from 'htm/preact';
import { render } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';

export function FileBrowserOpenBtn(props) {
	function handleClick() {
						
	}
	return html`
		<button onClick=${() => handleClick()} class='fileBrowserModalToggleBtn'>${props.buttonText}</button>
	`
}
