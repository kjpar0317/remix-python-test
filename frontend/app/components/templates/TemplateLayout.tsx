import Header from "~/components/layouts/common/Header";

export default function TemplateLayout({ children }: {
    children: React.ReactNode;
}) {
    return (
      <div className="w-full h-full flex justify-center">
        <Header />
        <main className="p-4 mt-[50px] w-full md:mt-[72px] md:w-11/12">{children}</main>
      </div>
    );
}