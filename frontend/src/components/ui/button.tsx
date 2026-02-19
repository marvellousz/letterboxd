import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
    "inline-flex items-center justify-center whitespace-nowrap text-sm font-bold ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 active:translate-x-0.5 active:translate-y-0.5 active:shadow-none cursor-pointer",
    {
        variants: {
            variant: {
                default: "bg-main text-black neobrutalism-border neobrutalism-shadow hover:-translate-x-0.5 hover:-translate-y-0.5 hover:neobrutalism-shadow-lg",
                secondary: "bg-secondary text-black neobrutalism-border neobrutalism-shadow hover:-translate-x-0.5 hover:-translate-y-0.5 hover:neobrutalism-shadow-lg",
                accent: "bg-accent text-black neobrutalism-border neobrutalism-shadow hover:-translate-x-0.5 hover:-translate-y-0.5 hover:neobrutalism-shadow-lg",
                outline: "bg-white text-black neobrutalism-border neobrutalism-shadow hover:-translate-x-0.5 hover:-translate-y-0.5 hover:neobrutalism-shadow-lg",
                ghost: "hover:bg-accent hover:text-accent-foreground",
                link: "text-primary underline-offset-4 hover:underline",
            },
            size: {
                default: "h-10 px-6 py-2",
                sm: "h-9 px-3",
                lg: "h-12 px-8 text-base",
                icon: "h-10 w-10",
            },
        },
        defaultVariants: {
            variant: "default",
            size: "default",
        },
    }
)

export interface ButtonProps
    extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
    asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant, size, asChild = false, ...props }, ref) => {
        return (
            <button
                className={cn(buttonVariants({ variant, size, className }))}
                ref={ref}
                {...props}
            />
        )
    }
)
Button.displayName = "Button"

export { Button, buttonVariants }
