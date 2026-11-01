// Flattens a tour screenshot (PNG with transparent window corners) onto the landing page's dark
// background and writes a resized JPEG: swift scripts/shot-jpeg.swift in.png out.jpg [width]
import AppKit

let args = CommandLine.arguments
guard args.count >= 3, let src = NSImage(contentsOfFile: args[1]) else {
    FileHandle.standardError.write(Data("usage: shot-jpeg.swift in.png out.jpg [width]\n".utf8))
    exit(2)
}
let width = args.count > 3 ? Int(args[3]) ?? 1568 : 1568
let scale = CGFloat(width) / src.size.width
let size = NSSize(width: CGFloat(width), height: (src.size.height * scale).rounded())
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
                           bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                           colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
rep.size = size
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
NSGraphicsContext.current?.imageInterpolation = .high
NSColor(srgbRed: 0x0a / 255, green: 0x0a / 255, blue: 0x1e / 255, alpha: 1).setFill()  // the page's window frame colour
NSRect(origin: .zero, size: size).fill()
src.draw(in: NSRect(origin: .zero, size: size), from: .zero, operation: .sourceOver, fraction: 1)
NSGraphicsContext.restoreGraphicsState()
guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else { exit(1) }
try! jpeg.write(to: URL(fileURLWithPath: args[2]))
