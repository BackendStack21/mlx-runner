// Renders the landing page's social card: swift scripts/og.swift site/assets/og.png
import AppKit
import CoreText

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "site/assets/og.png"
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
for f in ["SpaceGrotesk.woff2", "JetBrainsMono-Regular.woff2"] {
    CTFontManagerRegisterFontsForURL(root.appendingPathComponent("site/fonts/\(f)") as CFURL, .process, nil)
}
/// `wght` is the weight on Space Grotesk's variable axis (500 on the page's headings); nil leaves the font as it is.
func font(_ name: String, _ size: CGFloat, wght: Int? = nil) -> NSFont {
    guard let base = NSFont(name: name, size: size) else { return .systemFont(ofSize: size) }
    guard let wght else { return base }
    let axis = 0x77676874 // 'wght'
    return NSFont(descriptor: base.fontDescriptor.addingAttributes([.variation: [axis: wght]]), size: size) ?? base
}
let W = 1200, H = 630
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H, bitsPerSample: 8, samplesPerPixel: 4,
                           hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
// The 21no.de palette (site/site.css): near-black, one cyan accent. The app mark keeps the app's own colours.
let indigo = NSColor(red: 0.42, green: 0.36, blue: 1.0, alpha: 1), cyan = NSColor(red: 0.13, green: 0.73, blue: 0.98, alpha: 1)
let accent = NSColor(red: 0.459, green: 0.839, blue: 0.910, alpha: 1)
NSColor(red: 0.020, green: 0.020, blue: 0.027, alpha: 1).setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()
// One faint glow, top right, and the 44 px grid fading out to the left.
let glow = NSColor(red: 0.263, green: 0.600, blue: 0.694, alpha: 1)
NSGradient(colors: [glow.withAlphaComponent(0.22), glow.withAlphaComponent(0)])!
    .draw(fromCenter: NSPoint(x: 1040, y: 640), radius: 0, toCenter: NSPoint(x: 1040, y: 640), radius: 560, options: [])
for x in stride(from: 0, through: W, by: 44) {
    NSColor(red: 0.58, green: 0.64, blue: 0.72, alpha: 0.06 * CGFloat(x) / CGFloat(W)).setFill()
    NSRect(x: x, y: 0, width: 1, height: H).fill()
}
for y in stride(from: H, through: 0, by: -44) {
    NSGradient(colors: [NSColor(red: 0.58, green: 0.64, blue: 0.72, alpha: 0), NSColor(red: 0.58, green: 0.64, blue: 0.72, alpha: 0.06)])!
        .draw(in: NSRect(x: 0, y: y, width: W, height: 1), angle: 0)
}
// App mark.
let mark = NSRect(x: 80, y: 430, width: 110, height: 110)
NSGradient(colors: [indigo, cyan])!.draw(in: NSBezierPath(roundedRect: mark, xRadius: 28, yRadius: 28), angle: -45)
if let s = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil)?
    .withSymbolConfiguration(.init(pointSize: 58, weight: .semibold)) {
    let tinted = NSImage(size: s.size, flipped: false) { r in s.draw(in: r); NSColor.white.set(); r.fill(using: .sourceAtop); return true }
    tinted.draw(in: NSRect(x: mark.midX - s.size.width / 2, y: mark.midY - s.size.height / 2, width: s.size.width, height: s.size.height))
}
func text(_ s: String, _ f: NSFont, _ c: NSColor, _ x: CGFloat, _ y: CGFloat) {
    (s as NSString).draw(at: NSPoint(x: x, y: y), withAttributes: [.font: f, .foregroundColor: c])
}
text("MLX Runner", font("Space Grotesk", 46, wght: 500), .white, 215, 462)
text("LLMs at the speed", font("Space Grotesk", 84, wght: 500), .white, 76, 290)
let line2 = NSMutableAttributedString(string: "of unified memory.", attributes: [.font: font("Space Grotesk", 84, wght: 500), .foregroundColor: NSColor.white])
line2.addAttribute(.foregroundColor, value: accent, range: NSRange(location: 3, length: 15))
line2.draw(at: NSPoint(x: 76, y: 190))
text("Native macOS · 100% Swift · its own Metal engine · zero dependencies", font("JetBrains Mono", 22),
     NSColor.white.withAlphaComponent(0.72), 80, 120)
text("mlx-runner.21no.de", font("JetBrains Mono", 22), accent, 80, 58)
text("by 21no.de", font("JetBrains Mono", 22), NSColor.white.withAlphaComponent(0.55), 1000, 58)
NSGraphicsContext.current = nil
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
