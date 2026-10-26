"""Run the wunet video pipeline; use --help for options."""

if __package__:
    from .pipeline import main
else:
    from pipeline import main

if __name__ == '__main__':
    main(default_mode='wunet')
