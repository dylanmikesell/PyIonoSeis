document$.subscribe(() => {
    if (window.mermaid) {
        window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: "loose",
            theme: "default",
        });

        const mermaidBlocks = document.querySelectorAll("pre.mermaid");
        if (mermaidBlocks.length > 0) {
            window.mermaid.run({
                nodes: mermaidBlocks,
            });
        }
    }
});
